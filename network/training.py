"""Training loops and losses for Stage1 label VAE and Stage2 MeanFlow."""

import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from network.models import Denoiser, ImageEncoder, LabelDecoder, LabelEncoder


class DiceLoss(nn.Module):
    """LSFSeg multiclass Dice loss."""

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = int(n_classes)

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for class_index in range(self.n_classes):
            tensor_list.append((input_tensor == class_index).unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1.0 - (2.0 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1.0] * self.n_classes
        if inputs.size() != target.size():
            raise ValueError(
                f"predict {tuple(inputs.size())} & target {tuple(target.size())} shape do not match"
            )
        loss = 0.0
        for class_index in range(self.n_classes):
            loss = (
                loss
                + self._dice_loss(inputs[:, class_index], target[:, class_index])
                * weight[class_index]
            )
        return loss / self.n_classes


class MulticlassFocalLoss(nn.Module):
    """One-vs-rest multiclass focal loss using the paper's two-term formula."""

    def __init__(self, n_classes=11, alpha=0.25, gamma=2.0):
        super().__init__()
        self.n_classes = int(n_classes)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("focal alpha must be in [0, 1]")
        if self.gamma < 0.0:
            raise ValueError("focal gamma must be non-negative")

    def forward(self, logits, target):
        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = (
            F.one_hot(
                target.long(),
                num_classes=self.n_classes,
            )
            .permute(0, 3, 1, 2)
            .to(dtype=probabilities.dtype)
        )

        eps = torch.finfo(probabilities.dtype).eps
        probabilities = probabilities.clamp(min=eps, max=1.0 - eps)
        positive = (
            -self.alpha
            * (1.0 - probabilities).pow(self.gamma)
            * probabilities.log()
            * target_one_hot
        )
        negative = (
            -(1.0 - self.alpha)
            * probabilities.pow(self.gamma)
            * torch.log1p(-probabilities)
            * (1.0 - target_one_hot)
        )
        return (positive + negative).sum(dim=1).mean()


class ReconstructionLoss(nn.Module):
    """Stage-2 segmentation loss: focal_weight * focal + dice_weight * Dice."""

    def __init__(
        self,
        n_classes=11,
        focal_weight=1,
        dice_weight=2,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        super().__init__()
        self.focal = MulticlassFocalLoss(
            n_classes=n_classes,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
        self.dice = DiceLoss(n_classes=n_classes)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, prediction, target):
        focal_loss = self.focal(prediction, target)
        dice_loss = self.dice(prediction, target, softmax=True)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


class Stage1VAELoss(nn.Module):
    """LSFSeg label VAE loss: cross-entropy, soft Dice, and optional KL."""

    def __init__(self, n_classes=11, ce_weight=1.0, dice_weight=1.0, kl_weight=0.0):
        super().__init__()
        self.n_classes = int(n_classes)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.kl_weight = float(kl_weight)

    def soft_dice_loss(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target.long(), num_classes=self.n_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_oh, dims)
        denominator = torch.sum(probs + target_oh, dims)
        dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        return 1.0 - dice.mean()

    def forward(self, logits, target, kl_loss=None):
        ce = F.cross_entropy(logits, target.long())
        dice = self.soft_dice_loss(logits, target)
        if kl_loss is None:
            kl_loss = logits.new_zeros(())
        loss = self.ce_weight * ce + self.dice_weight * dice + self.kl_weight * kl_loss
        return loss, {"ce": ce, "dice": dice, "kl": kl_loss}


def build_warmup_linear_scheduler(
    optimizer, total_steps, warmup_steps=10000, min_lr_scale=0.0
):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_lr_scale = float(min_lr_scale)

    def lr_lambda(step):
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            return max(min_lr_scale, float(step + 1) / float(warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        return max(min_lr_scale, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class LSFSegTrainer:
    """
    Latent label-condition trainer using pixel MeanFlow-style denoising.

    MeanFlow uses t=0 for clean data and t=1 for noise:
        z_t = (1 - t) * x + t * eps
        v_t = (z_t - x) / t
    The denoiser predicts the average velocity u and an auxiliary
    instantaneous velocity v in latent space, conditioned on image_latent.
    """

    def __init__(
        self,
        network,
        ema_network,
        img_encoder,
        encoder,
        decoder,
        filepath,
        num_classes=11,
        ema_decay=0.999,
        segmentation_weight=1.0,
        denoise_weight=1.0,
        p_mean=-0.8,
        p_std=0.8,
        noise_scale=1.0,
        t_eps=0.05,
        meanflow_data_proportion=0.5,
        meanflow_norm_p=1.0,
        meanflow_norm_eps=0.01,
        dual_view_consistency_weight=0.05,
        dual_view_latent_weight=0.25,
        dual_view_warmup_epochs=10,
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.network = network.to(self.device)
        self.ema_network = ema_network.to(self.device)
        self.img_encoder = img_encoder.to(self.device)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.filepath = filepath
        self.num_classes = int(num_classes)
        self.ema_decay = float(ema_decay)
        self.segmentation_weight = float(segmentation_weight)
        self.denoise_weight = float(denoise_weight)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.noise_scale = float(noise_scale)
        self.t_eps = float(t_eps)
        self.meanflow_data_proportion = float(meanflow_data_proportion)
        self.meanflow_norm_p = float(meanflow_norm_p)
        self.meanflow_norm_eps = float(meanflow_norm_eps)
        self.dual_view_consistency_weight = float(dual_view_consistency_weight)
        self.dual_view_latent_weight = float(dual_view_latent_weight)
        self.dual_view_warmup_epochs = int(dual_view_warmup_epochs)
        if not 0.0 <= self.meanflow_data_proportion <= 1.0:
            raise ValueError("meanflow_data_proportion must be in [0, 1]")
        if self.segmentation_weight < 0.0:
            raise ValueError("segmentation_weight must be non-negative")
        if self.dual_view_consistency_weight < 0.0:
            raise ValueError("dual_view_consistency_weight must be non-negative")
        if self.dual_view_latent_weight < 0.0:
            raise ValueError("dual_view_latent_weight must be non-negative")
        if self.dual_view_warmup_epochs < 0:
            raise ValueError("dual_view_warmup_epochs must be non-negative")
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(True)
        os.makedirs(filepath, exist_ok=True)
        self._copy_to_ema()

    def _copy_to_ema(self):
        self.ema_network.load_state_dict(self.network.state_dict())
        self.ema_network.requires_grad_(False)

    @torch.no_grad()
    def _update_ema(self):
        for ema_param, param in zip(
            self.ema_network.parameters(), self.network.parameters()
        ):
            ema_param.mul_(self.ema_decay).add_(param, alpha=1.0 - self.ema_decay)

    def sample_t(self, n: int, device=None):
        """Sample MeanFlow time from a sigmoid-normal distribution."""
        z = torch.randn(n, device=device or self.device) * self.p_std + self.p_mean
        return torch.sigmoid(z)

    def sample_tr(self, n: int, device=None):
        """Sample MeanFlow (t, r) with t >= r; r=t gives FM samples."""
        device = device or self.device
        t = self.sample_t(n, device=device)
        r = self.sample_t(n, device=device)
        fm_mask = torch.rand(n, device=device) < self.meanflow_data_proportion
        r = torch.where(fm_mask, t, r)
        t_max = torch.maximum(t, r)
        r_min = torch.minimum(t, r)
        return t_max, r_min, fm_mask

    def make_meanflow_target(self, x):
        """MeanFlow target in latent space; t=0 is clean, t=1 is noise."""
        t_flat, r_flat, _ = self.sample_tr(x.size(0), device=x.device)
        t = t_flat.view(
            -1,
            *([1] * (x.ndim - 1)),
        )
        e = torch.randn_like(x) * self.noise_scale
        z = (1.0 - t) * x + t * e
        v = (z - x) / t.clamp_min(self.t_eps)
        return z, v, t_flat, r_flat

    def meanflow_adaptive_loss(self, pred, target):
        per_sample = (pred - target).pow(2).flatten(1).sum(dim=1)
        weight = (per_sample.detach() + self.meanflow_norm_eps).pow(
            self.meanflow_norm_p
        )
        return (per_sample / weight.clamp_min(1e-12)).mean()

    def _save_model_tag(self, tag):
        torch.save(
            self.encoder.state_dict(),
            os.path.join(self.filepath, f"labelEncoder_{tag}.pt"),
        )
        torch.save(
            self.decoder.state_dict(),
            os.path.join(self.filepath, f"labelDecoder_{tag}.pt"),
        )
        torch.save(
            self.img_encoder.state_dict(),
            os.path.join(self.filepath, f"imageEncoder_{tag}.pt"),
        )
        torch.save(
            self.ema_network.state_dict(),
            os.path.join(self.filepath, f"denoiser_{tag}.pt"),
        )
        print(f"Models saved at {tag} to {self.filepath}")

    def _remove_non_best_model_weights(self):
        """Keep only the four files belonging to the best Stage-2 checkpoint."""
        prefixes = ("labelEncoder", "labelDecoder", "imageEncoder", "denoiser")
        best_filenames = {f"{prefix}_best_epoch.pt" for prefix in prefixes}
        removed = []
        for entry in os.scandir(self.filepath):
            if not entry.is_file() or entry.name in best_filenames:
                continue
            if entry.name.endswith(".pt") and any(
                entry.name.startswith(f"{prefix}_") for prefix in prefixes
            ):
                os.remove(entry.path)
                removed.append(entry.name)
        if removed:
            print(
                f"Removed {len(removed)} superseded Stage-2 weight file(s).",
                flush=True,
            )

    def save_best_model(self, epoch, validation_segmentation_loss):
        """Overwrite the single best checkpoint group and remove older groups."""
        self._save_model_tag("best_epoch")
        self._remove_non_best_model_weights()
        print(
            "Updated best Stage-2 checkpoint at "
            f"epoch {int(epoch)}: validation segmentation loss="
            f"{float(validation_segmentation_loss):.8f}",
            flush=True,
        )

    def dual_view_weight_for_epoch(self, epoch):
        """Linearly warm up the Stage-2 consistency contribution."""
        if self.dual_view_warmup_epochs <= 0:
            return self.dual_view_consistency_weight
        scale = min(1.0, max(0.0, float(epoch)) / float(self.dual_view_warmup_epochs))
        return self.dual_view_consistency_weight * scale

    def _meanflow_branch(
        self,
        denoiser,
        z,
        v_t,
        t_flat,
        r_flat,
        image_latent,
    ):
        # Keep the complete four-channel image condition for every sample.
        u, v_pred = denoiser(
            z,
            image_latent,
            t_flat,
            h_input=t_flat - r_flat,
            return_velocity=True,
        )
        denoising = self.meanflow_adaptive_loss(u, v_t) + self.meanflow_adaptive_loss(
            v_pred, v_t
        )
        return denoising, u

    def _forward_batch(self, sample, denoiser, reconstruction_loss, epoch=0):
        image = sample["image"].to(self.device)
        image_strong = sample.get("image_strong")
        if image_strong is not None:
            image_strong = image_strong.to(self.device)
        label = sample["label"].to(self.device)
        label_condition = sample["sdf"].to(self.device)
        use_dual_view = denoiser.training and image_strong is not None

        with torch.no_grad():
            clean_latent = self.encoder(label_condition)
        image_latent, image_skip_features = self.img_encoder(
            image,
            return_skips=True,
        )
        if use_dual_view:
            image_latent_strong, _image_skip_features_strong = self.img_encoder(
                image_strong,
                return_skips=True,
            )
        else:
            image_latent_strong = None
        deepest_image_feature = image_skip_features[-1]
        z, v_t, t_flat, r_flat = self.make_meanflow_target(clean_latent)

        denoising_weak, u_weak = self._meanflow_branch(
            denoiser=denoiser,
            z=z,
            v_t=v_t,
            t_flat=t_flat,
            r_flat=r_flat,
            image_latent=image_latent,
        )
        velocity_consistency = denoising_weak.new_zeros(())
        latent_consistency = denoising_weak.new_zeros(())
        dual_view_consistency = denoising_weak.new_zeros(())
        consistency_weight = denoising_weak.new_tensor(
            self.dual_view_weight_for_epoch(epoch)
        )
        if use_dual_view:
            denoising_strong, u_strong = self._meanflow_branch(
                denoiser=denoiser,
                z=z,
                v_t=v_t,
                t_flat=t_flat,
                r_flat=r_flat,
                image_latent=image_latent_strong,
            )
            denoising = 0.5 * (denoising_weak + denoising_strong)
            h = (t_flat - r_flat).view(-1, *([1] * (clean_latent.ndim - 1)))
            z_hat_weak = z - h * u_weak
            z_hat_strong = z - h * u_strong
            velocity_consistency = F.smooth_l1_loss(
                u_strong,
                u_weak.detach(),
                reduction="mean",
            )
            latent_consistency = F.smooth_l1_loss(
                z_hat_strong,
                z_hat_weak.detach(),
                reduction="mean",
            )
            dual_view_consistency = (
                velocity_consistency + self.dual_view_latent_weight * latent_consistency
            )
        else:
            denoising = denoising_weak
        t = t_flat.view(-1, *([1] * (clean_latent.ndim - 1)))
        x_pred = z - t * u_weak
        label_logits = self.decoder(
            x_pred,
            image_skip_features,
            image_latent_feature=deepest_image_feature,
        )

        reconstruction = reconstruction_loss(label_logits, label)
        total = (
            self.segmentation_weight * reconstruction
            + self.denoise_weight * denoising
            + consistency_weight * dual_view_consistency
        )

        return (
            total,
            reconstruction,
            denoising,
            dual_view_consistency,
            velocity_consistency,
            latent_consistency,
            consistency_weight,
        )

    def train_epoch(self, dataloader, optimizers, reconstruction_loss, epoch):
        self.network.train()
        self.encoder.eval()
        self.decoder.train()
        self.img_encoder.train()
        self.ema_network.eval()

        totals = {
            "loss": 0.0,
            "loss1": 0.0,
            "loss2": 0.0,
            "dual_view_consistency": 0.0,
            "velocity_consistency": 0.0,
            "latent_consistency": 0.0,
            "consistency_weight": 0.0,
        }
        count = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", ncols=110)
        for sample in loop:
            batch_size = sample["image"].shape[0]
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

            (
                loss,
                reconstruction,
                velocity,
                dual_view_consistency,
                velocity_consistency,
                latent_consistency,
                consistency_weight,
            ) = self._forward_batch(
                sample,
                self.network,
                reconstruction_loss,
                epoch=epoch,
            )
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            self._update_ema()

            totals["loss"] += loss.item() * batch_size
            totals["loss1"] += reconstruction.item() * batch_size
            totals["loss2"] += velocity.item() * batch_size
            totals["dual_view_consistency"] += dual_view_consistency.item() * batch_size
            totals["velocity_consistency"] += velocity_consistency.item() * batch_size
            totals["latent_consistency"] += latent_consistency.item() * batch_size
            totals["consistency_weight"] += consistency_weight.item() * batch_size
            count += batch_size
            loop.set_postfix(loss=totals["loss"] / count)

        return {key: value / max(1, count) for key, value in totals.items()}

    @torch.no_grad()
    def validate_epoch(self, dataloader, reconstruction_loss, epoch=0):
        self.network.eval()
        self.ema_network.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.img_encoder.eval()

        totals = {
            "loss": 0.0,
            "loss1": 0.0,
            "loss2": 0.0,
            "dual_view_consistency": 0.0,
            "velocity_consistency": 0.0,
            "latent_consistency": 0.0,
            "consistency_weight": 0.0,
        }
        count = 0
        loop = tqdm(dataloader, desc="Validation", ncols=110)
        for sample in loop:
            batch_size = sample["image"].shape[0]
            (
                loss,
                reconstruction,
                velocity,
                dual_view_consistency,
                velocity_consistency,
                latent_consistency,
                consistency_weight,
            ) = self._forward_batch(
                sample,
                self.ema_network,
                reconstruction_loss,
                epoch=epoch,
            )
            totals["loss"] += loss.item() * batch_size
            totals["loss1"] += reconstruction.item() * batch_size
            totals["loss2"] += velocity.item() * batch_size
            totals["dual_view_consistency"] += dual_view_consistency.item() * batch_size
            totals["velocity_consistency"] += velocity_consistency.item() * batch_size
            totals["latent_consistency"] += latent_consistency.item() * batch_size
            totals["consistency_weight"] += consistency_weight.item() * batch_size
            count += batch_size
            loop.set_postfix(loss=totals["loss"] / count)
        return {key: value / max(1, count) for key, value in totals.items()}


def _save_label_vae(label_encoder, label_decoder, model_directory, tag):
    os.makedirs(model_directory, exist_ok=True)
    torch.save(
        label_encoder.state_dict(),
        os.path.join(model_directory, f"labelEncoder_{tag}.pt"),
    )
    torch.save(
        label_decoder.state_dict(),
        os.path.join(model_directory, f"labelDecoder_{tag}.pt"),
    )
    print(f"Label VAE saved at {tag} to {model_directory}")


def _metric(metrics, key):
    return float(metrics.get(key, 0.0))


def _write_stage2_tensorboard_metrics(writer, split, metrics, trainer, epoch):
    """Write explicit Stage-2 TensorBoard tags with raw and weighted losses."""

    reconstruction = _metric(metrics, "loss1")
    denoising = _metric(metrics, "loss2")
    dual_view_consistency = _metric(metrics, "dual_view_consistency")
    velocity_consistency = _metric(metrics, "velocity_consistency")
    latent_consistency = _metric(metrics, "latent_consistency")
    consistency_weight = _metric(metrics, "consistency_weight")

    weighted_denoising = trainer.denoise_weight * denoising
    weighted_reconstruction = trainer.segmentation_weight * reconstruction
    weighted_dual_view_consistency = consistency_weight * dual_view_consistency

    writer.add_scalar(f"{split}/Loss/total", _metric(metrics, "loss"), epoch)
    writer.add_scalar(f"{split}/Loss/seg_reconstruction_raw", reconstruction, epoch)
    writer.add_scalar(
        f"{split}/Loss/seg_reconstruction_weighted",
        weighted_reconstruction,
        epoch,
    )
    writer.add_scalar(f"{split}/Loss/meanflow_denoising_raw", denoising, epoch)
    writer.add_scalar(
        f"{split}/Loss/meanflow_denoising_weighted", weighted_denoising, epoch
    )
    writer.add_scalar(
        f"{split}/Loss/dual_view_consistency_raw",
        dual_view_consistency,
        epoch,
    )
    writer.add_scalar(
        f"{split}/Loss/dual_view_consistency_weighted",
        weighted_dual_view_consistency,
        epoch,
    )
    writer.add_scalar(
        f"{split}/DualView/velocity_consistency",
        velocity_consistency,
        epoch,
    )
    writer.add_scalar(
        f"{split}/DualView/latent_consistency",
        latent_consistency,
        epoch,
    )

    writer.add_scalar(f"{split}/Weights/denoise_weight", trainer.denoise_weight, epoch)
    writer.add_scalar(
        f"{split}/Weights/segmentation_weight",
        trainer.segmentation_weight,
        epoch,
    )
    writer.add_scalar(
        f"{split}/Weights/dual_view_consistency_weight_effective",
        consistency_weight,
        epoch,
    )
    writer.add_scalar(
        f"{split}/Weights/dual_view_latent_weight",
        trainer.dual_view_latent_weight,
        epoch,
    )


def _label_vae_epoch(
    label_encoder,
    label_decoder,
    dataloader,
    loss_fn,
    device,
    optimizer=None,
    scheduler=None,
    grad_clip=1.0,
    epoch=0,
    global_step=0,
    max_steps=None,
    save_step_callback=None,
):
    training = optimizer is not None
    label_encoder.train(training)
    label_decoder.train(training)
    totals = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "kl": 0.0, "acc": 0.0}
    foreground_dice_sum = None
    foreground_dice_count = 0
    count = 0
    desc = (
        f"Stage1 Label VAE Epoch {epoch} [Train]"
        if training
        else "Stage1 Label VAE Validation"
    )
    loop = tqdm(dataloader, desc=desc, ncols=110)

    for sample in loop:
        if training and max_steps is not None and int(global_step) >= int(max_steps):
            break
        label = sample["label"].to(device)
        label_condition = sample["sdf"].to(device)
        batch_size = int(label.shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            clean_latent = label_encoder(label_condition)
            logits = label_decoder(clean_latent, None)
            kl_loss = 0.5 * clean_latent.pow(2).flatten(1).sum(dim=1).mean()
            kl_loss = kl_loss / float(max(1, label_condition[0].numel()))
            loss, pieces = loss_fn(logits, label, kl_loss=kl_loss)
            if training:
                loss.backward()
                if grad_clip is not None and float(grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(label_encoder.parameters())
                        + list(label_decoder.parameters()),
                        float(grad_clip),
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if save_step_callback is not None:
                    save_step_callback(global_step)

        prediction = torch.argmax(logits.detach(), dim=1)
        acc = (prediction == label).float().mean()
        foreground_classes = int(logits.shape[1]) - 1
        if foreground_dice_sum is None:
            foreground_dice_sum = torch.zeros(
                (), dtype=torch.float64, device=prediction.device
            )
        for class_id in range(1, foreground_classes + 1):
            predicted_class = prediction == class_id
            target_class = label == class_id
            intersections = (
                torch.logical_and(
                    predicted_class,
                    target_class,
                )
                .flatten(1)
                .sum(dim=1)
                .to(torch.float64)
            )
            denominators = (
                predicted_class.flatten(1).sum(dim=1)
                + target_class.flatten(1).sum(dim=1)
            ).to(torch.float64)
            foreground_dice_sum += torch.where(
                denominators > 0,
                2.0 * intersections / denominators.clamp_min(1.0),
                torch.zeros_like(denominators),
            ).sum()
            foreground_dice_count += batch_size
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["ce"] += float(pieces["ce"].detach().cpu()) * batch_size
        totals["dice"] += float(pieces["dice"].detach().cpu()) * batch_size
        totals["kl"] += float(pieces["kl"].detach().cpu()) * batch_size
        totals["acc"] += float(acc.detach().cpu()) * batch_size
        count += batch_size
        loop.set_postfix(
            loss=totals["loss"] / max(1, count), acc=totals["acc"] / max(1, count)
        )

    metrics = {key: value / max(1, count) for key, value in totals.items()}
    if foreground_dice_sum is None:
        metrics["mean_foreground_dice"] = 0.0
    else:
        metrics["mean_foreground_dice"] = float(
            foreground_dice_sum / max(1, foreground_dice_count)
        )
    return metrics, global_step


def train_label_vae(
    labelEncoder: LabelEncoder,
    labelDecoder: LabelDecoder,
    train_dataset: Dataset,
    val_dataset: Dataset,
    epochs=300,
    batch_size=4,
    lr=1e-5,
    modelDirectory="./stage1",
    device=None,
    validate_every=1,
    save_every=10,
    num_classes=11,
    ce_weight=1.0,
    dice_weight=1.0,
    kl_weight=0.0,
    warmup_steps=10000,
    lr_scheduler="warmup_linear",
    min_lr_scale=0.0,
    grad_clip=1.0,
    max_steps=None,
    save_interval_steps=0,
):
    """Stage-1 pure label VAE/AE training without image input or denoiser."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labelEncoder = labelEncoder.to(device)
    labelDecoder = labelDecoder.to(device)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    writer = SummaryWriter(os.path.join(modelDirectory, "tensorboard_logs"))
    optimizer = torch.optim.AdamW(
        list(labelEncoder.parameters()) + list(labelDecoder.parameters()),
        lr=lr,
    )
    total_steps = max(
        1,
        int(max_steps)
        if max_steps is not None
        else int(epochs) * max(1, len(train_loader)),
    )
    scheduler = None
    if str(lr_scheduler).lower() == "warmup_linear":
        scheduler = build_warmup_linear_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_scale=min_lr_scale,
        )
    elif str(lr_scheduler).lower() != "constant":
        raise ValueError("lr_scheduler must be 'constant' or 'warmup_linear'")

    loss_fn = Stage1VAELoss(
        n_classes=num_classes,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        kl_weight=kl_weight,
    )
    best_val_loss = float("inf")
    global_step = 0

    def save_step_checkpoint(step):
        interval = int(save_interval_steps or 0)
        if interval > 0 and int(step) % interval == 0:
            _save_label_vae(
                labelEncoder, labelDecoder, modelDirectory, f"step_{int(step)}"
            )

    for epoch in tqdm(
        range(1, epochs + 1), desc="Stage1 Label VAE Training", ncols=110
    ):
        if max_steps is not None and global_step >= int(max_steps):
            break
        train_metrics, global_step = _label_vae_epoch(
            labelEncoder,
            labelDecoder,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=grad_clip,
            epoch=epoch,
            global_step=global_step,
            max_steps=max_steps,
            save_step_callback=save_step_checkpoint,
        )
        for key, value in train_metrics.items():
            writer.add_scalar(f"Stage1/Train/{key}", value, epoch)
        writer.add_scalar("Stage1/Train/lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Stage1/Train/global_step", global_step, epoch)

        if epoch % int(validate_every) == 0:
            with torch.no_grad():
                val_metrics, _ = _label_vae_epoch(
                    labelEncoder,
                    labelDecoder,
                    val_loader,
                    loss_fn,
                    device,
                    optimizer=None,
                    epoch=epoch,
                    global_step=global_step,
                )
            for key, value in val_metrics.items():
                writer.add_scalar(f"Stage1/Val/{key}", value, epoch)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                _save_label_vae(
                    labelEncoder, labelDecoder, modelDirectory, "best_epoch"
                )

        if save_every > 0 and epoch % int(save_every) == 0:
            _save_label_vae(
                labelEncoder, labelDecoder, modelDirectory, f"epoch_{epoch}"
            )

        if max_steps is not None and global_step >= int(max_steps):
            break

    _save_label_vae(labelEncoder, labelDecoder, modelDirectory, "last_epoch")
    writer.close()


def train_models(
    labelEncoder: LabelEncoder,
    labelDecoder: LabelDecoder,
    imageEncoder: ImageEncoder,
    denoiser: Denoiser,
    train_dataset: Dataset,
    val_dataset: Dataset,
    epochs=1500,
    start_epoch=1,
    batch_size=4,
    lr=1e-3,
    lr_decay_epochs=0,
    lr_decay_gamma=0.5,
    modelDirectory="./stage2",
    device=None,
    validate_every=1,
    early_stop_patience=30,
    enable_early_stop=False,
    num_classes=11,
    p_mean=-0.8,
    p_std=0.8,
    noise_scale=1.0,
    t_eps=0.05,
    ema_decay=0.999,
    segmentation_weight=1.0,
    denoise_weight=1.0,
    meanflow_data_proportion=0.5,
    meanflow_norm_p=1.0,
    meanflow_norm_eps=0.01,
    dual_view_consistency_weight=0.05,
    dual_view_latent_weight=0.25,
    dual_view_warmup_epochs=10,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_epoch = int(start_epoch)
    epochs = int(epochs)
    lr_decay_epochs = int(lr_decay_epochs)
    lr_decay_gamma = float(lr_decay_gamma)
    if start_epoch < 1:
        raise ValueError("start_epoch must be at least 1")
    if start_epoch > epochs:
        raise ValueError(
            f"start_epoch={start_epoch} is greater than max epoch={epochs}; "
            "increase --max_epochs or select an earlier Stage-2 checkpoint"
        )
    if lr_decay_epochs < 0:
        raise ValueError("lr_decay_epochs must be non-negative")
    if not 0.0 < lr_decay_gamma <= 1.0:
        raise ValueError("lr_decay_gamma must be in (0, 1]")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    writer = SummaryWriter(
        os.path.join(modelDirectory, "tensorboard_logs"),
        purge_step=start_epoch if start_epoch > 1 else None,
    )

    network = denoiser.to(device)
    ema_network = copy.deepcopy(denoiser).to(device)
    trainer = LSFSegTrainer(
        network=network,
        ema_network=ema_network,
        img_encoder=imageEncoder,
        encoder=labelEncoder,
        decoder=labelDecoder,
        filepath=modelDirectory,
        num_classes=num_classes,
        ema_decay=ema_decay,
        segmentation_weight=segmentation_weight,
        denoise_weight=denoise_weight,
        p_mean=p_mean,
        p_std=p_std,
        noise_scale=noise_scale,
        t_eps=t_eps,
        meanflow_data_proportion=meanflow_data_proportion,
        meanflow_norm_p=meanflow_norm_p,
        meanflow_norm_eps=meanflow_norm_eps,
        dual_view_consistency_weight=dual_view_consistency_weight,
        dual_view_latent_weight=dual_view_latent_weight,
        dual_view_warmup_epochs=dual_view_warmup_epochs,
        device=device,
    )

    # Use one Adam optimizer per trainable module with a shared learning rate.
    optimizers = []
    for module in [
        trainer.network,
        trainer.encoder,
        trainer.decoder,
        trainer.img_encoder,
    ]:
        params = [
            parameter for parameter in module.parameters() if parameter.requires_grad
        ]
        if params:
            optimizers.append(torch.optim.Adam(params, lr=lr))
    if not optimizers:
        raise ValueError("No trainable parameters were found for stage-2 training.")
    lr_schedulers = (
        [
            StepLR(
                optimizer,
                step_size=lr_decay_epochs,
                gamma=lr_decay_gamma,
            )
            for optimizer in optimizers
        ]
        if lr_decay_epochs > 0
        else []
    )
    reconstruction_loss = ReconstructionLoss(
        n_classes=num_classes,
        focal_weight=1,
        dice_weight=2,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ).to(device)
    best_val_segmentation_loss = float("inf")
    best_epoch = None
    patience = 0

    for epoch in tqdm(
        range(start_epoch, epochs + 1),
        desc="Training Epochs",
        ncols=110,
    ):
        current_lr = float(optimizers[0].param_groups[0]["lr"])
        writer.add_scalar("Train/learning_rate", current_lr, epoch)
        train_metrics = trainer.train_epoch(
            train_loader,
            optimizers,
            reconstruction_loss,
            epoch,
        )
        for key, value in train_metrics.items():
            writer.add_scalar(f"Train/{key}", value, epoch)
        writer.add_scalar("Train/total_loss", train_metrics["loss"], epoch)
        writer.add_scalar("Train/segmentation_loss", train_metrics["loss1"], epoch)
        writer.add_scalar("Train/flow_loss", train_metrics["loss2"], epoch)
        _write_stage2_tensorboard_metrics(
            writer, "Train", train_metrics, trainer, epoch
        )

        if epoch % validate_every == 0:
            val_metrics = trainer.validate_epoch(
                val_loader,
                reconstruction_loss,
                epoch=epoch,
            )
            for key, value in val_metrics.items():
                writer.add_scalar(f"Val/{key}", value, epoch)
            writer.add_scalar("Val/total_loss", val_metrics["loss"], epoch)
            writer.add_scalar("Val/segmentation_loss", val_metrics["loss1"], epoch)
            writer.add_scalar("Val/flow_loss", val_metrics["loss2"], epoch)
            _write_stage2_tensorboard_metrics(
                writer, "Val", val_metrics, trainer, epoch
            )

            val_segmentation_loss = float(val_metrics["loss1"])
            if (
                math.isfinite(val_segmentation_loss)
                and val_segmentation_loss < best_val_segmentation_loss
            ):
                best_val_segmentation_loss = val_segmentation_loss
                best_epoch = epoch
                patience = 0
                trainer.save_best_model(epoch, val_segmentation_loss)
            else:
                patience += 1

            if enable_early_stop and patience >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        for scheduler in lr_schedulers:
            scheduler.step()
        completed_epochs = epoch - start_epoch + 1
        if lr_schedulers and completed_epochs % lr_decay_epochs == 0:
            next_lr = float(optimizers[0].param_groups[0]["lr"])
            print(
                f"Stage-2 learning rate decayed after {completed_epochs} resumed "
                f"epoch(s): {current_lr:.8g} -> {next_lr:.8g}",
                flush=True,
            )

    writer.close()
    if best_epoch is None:
        raise RuntimeError(
            "Stage-2 training completed without a finite validation segmentation "
            "loss, so no best checkpoint could be saved."
        )
    print(
        "Stage-2 training complete. Retained only best_epoch from "
        f"epoch {best_epoch} with validation segmentation loss "
        f"{best_val_segmentation_loss:.8f}.",
        flush=True,
    )
