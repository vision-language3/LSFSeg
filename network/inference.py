"""Inference, sampling, metric, and visualization routines for MeanFlow testing."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
from tqdm import tqdm

from network.config import CLASS_COLORS, CLASS_NAMES
from network.models import load_denoiser_state_dict, load_label_decoder_state_dict
from network.utils import add_noise, plot_sampling

MGU_CLASS_NAMES = CLASS_NAMES


def dice_coefficient(pred, true):
    intersection = np.logical_and(pred, true).sum()
    size_sum = pred.sum() + true.sum()
    if size_sum == 0:
        return 1.0
    return 2.0 * intersection / size_sum


def hd95(pred, true):
    pred_border = pred ^ binary_erosion(pred)
    true_border = true ^ binary_erosion(true)
    pred_points = np.argwhere(pred_border)
    true_points = np.argwhere(true_border)
    if len(pred_points) == 0 or len(true_points) == 0:
        return 0.0
    dists1 = cdist(pred_points, true_points).min(axis=1)
    dists2 = cdist(true_points, pred_points).min(axis=1)
    return np.percentile(np.hstack([dists1, dists2]), 95)


def calculate_multiclass_metrics(y_true_np, y_pred_np, num_classes=11, ignore_index=0):
    """Keep LSFSeg's per-class Dice, HD95, and IoU behavior."""
    dice_list, hd95_list, iou_list = [], [], []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_cls = (y_pred_np == cls).astype(np.uint8)
        true_cls = (y_true_np == cls).astype(np.uint8)

        if pred_cls.sum() > 0 and true_cls.sum() > 0:
            dice = dice_coefficient(pred_cls, true_cls)
            hd95_val = hd95(pred_cls, true_cls)
            intersection = np.logical_and(pred_cls, true_cls).sum()
            union = np.logical_or(pred_cls, true_cls).sum()
            iou = intersection / union if union > 0 else 0
        elif pred_cls.sum() > 0 and true_cls.sum() == 0:
            # A predicted region with no corresponding ground truth is a
            # false positive, not a perfect Dice score.
            dice, hd95_val, iou = 0, 0, 0
        else:
            dice, hd95_val, iou = 0, 0, 0

        dice_list.append(dice)
        hd95_list.append(hd95_val)
        iou_list.append(iou)
    return dice_list, hd95_list, iou_list


def aggregate_metrics(confusion_matrix, all_hd95):
    """Aggregate foreground global-pixel Dice/IoU and per-case HD95."""
    confusion_matrix = np.asarray(confusion_matrix, dtype=np.float64)
    hd95_arr = np.asarray(all_hd95, dtype=np.float64)
    if (
        confusion_matrix.ndim != 2
        or confusion_matrix.shape[0] != confusion_matrix.shape[1]
    ):
        raise ValueError(
            "MGU global metric aggregation requires a square confusion matrix, "
            f"got shape {confusion_matrix.shape}"
        )
    num_classes = confusion_matrix.shape[0]
    if hd95_arr.ndim != 2 or hd95_arr.shape[1] != num_classes - 1:
        raise ValueError(
            "MGU HD95 aggregation requires one value per foreground class, "
            f"got shape {hd95_arr.shape}"
        )

    true_pixels = confusion_matrix.sum(axis=1)
    predicted_pixels = confusion_matrix.sum(axis=0)
    true_positive = np.diag(confusion_matrix)
    false_negative = true_pixels - true_positive
    false_positive = predicted_pixels - true_positive

    dice_denominator = 2.0 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    global_dice_per_class = np.divide(
        2.0 * true_positive,
        dice_denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=dice_denominator > 0,
    )
    global_iou_per_class = np.divide(
        true_positive,
        iou_denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=iou_denominator > 0,
    )
    mean_hd95_per_class = hd95_arr.mean(axis=0)
    foreground_dice = global_dice_per_class[1:]
    foreground_iou = global_iou_per_class[1:]
    return {
        "dice_per_class": foreground_dice.tolist(),
        "hd95_per_class": mean_hd95_per_class.tolist(),
        "iou_per_class": foreground_iou.tolist(),
        "mean_dice_per_class": foreground_dice.tolist(),
        "mean_dice": float(foreground_dice.mean()),
        "mean_hd95": float(mean_hd95_per_class.mean()),
        "mean_iou": float(foreground_iou.mean()),
        "mean_dice_all_10": float(foreground_dice.mean()),
        "mean_hd95_all_10": float(mean_hd95_per_class.mean()),
        "mean_iou_all_10": float(foreground_iou.mean()),
        "mean_dice_retina_9_no_optic_disc": float(foreground_dice[:9].mean()),
        "mean_hd95_retina_9_no_optic_disc": float(mean_hd95_per_class[:9].mean()),
        "mean_iou_retina_9_no_optic_disc": float(foreground_iou[:9].mean()),
    }


def calculate_confusion_matrix(y_true_np, y_pred_np, num_classes=11):
    """Build a true-class rows, predicted-class columns confusion matrix."""
    true_flat = np.asarray(y_true_np).reshape(-1)
    pred_flat = np.asarray(y_pred_np).reshape(-1)
    valid_mask = (
        (true_flat >= 0)
        & (true_flat < num_classes)
        & (pred_flat >= 0)
        & (pred_flat < num_classes)
    )
    if not valid_mask.any():
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    bins = num_classes * true_flat[valid_mask].astype(np.int64) + pred_flat[
        valid_mask
    ].astype(np.int64)
    return np.bincount(
        bins,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def summarize_sample_metrics(
    case_name,
    y_true_np,
    y_pred_np,
    hd95_per_class,
    num_classes=11,
):
    """Summarize one sample with foreground-only global-pixel metrics."""
    sample_confusion = calculate_confusion_matrix(
        y_true_np,
        y_pred_np,
        num_classes=num_classes,
    ).astype(np.float64)
    true_pixels = sample_confusion.sum(axis=1)
    predicted_pixels = sample_confusion.sum(axis=0)
    true_positive = np.diag(sample_confusion)
    false_negative = true_pixels - true_positive
    false_positive = predicted_pixels - true_positive

    foreground_slice = slice(1, None)
    foreground_tp = true_positive[foreground_slice]
    foreground_fp = false_positive[foreground_slice]
    foreground_fn = false_negative[foreground_slice]

    dice_denominator = 2.0 * foreground_tp + foreground_fp + foreground_fn
    iou_denominator = foreground_tp + foreground_fp + foreground_fn
    dice_per_class = np.divide(
        2.0 * foreground_tp,
        dice_denominator,
        out=np.zeros_like(foreground_tp, dtype=np.float64),
        where=dice_denominator > 0,
    )
    iou_per_class = np.divide(
        foreground_tp,
        iou_denominator,
        out=np.zeros_like(foreground_tp, dtype=np.float64),
        where=iou_denominator > 0,
    )
    hd95_arr = np.asarray(hd95_per_class, dtype=np.float64)
    return {
        "case_name": str(case_name),
        "mean_dice_all_10": float(dice_per_class.mean()),
        "mean_iou_all_10": float(iou_per_class.mean()),
        "mean_hd95_all_10": float(hd95_arr.mean()),
        "false_positive_pixels": int(foreground_fp.sum()),
        "false_negative_pixels": int(foreground_fn.sum()),
        "foreground_gt_pixels": int(true_pixels[foreground_slice].sum()),
        "foreground_pred_pixels": int(predicted_pixels[foreground_slice].sum()),
    }


def _forward_sample(
    denoiser,
    z,
    image_latent,
    t,
    t_eps,
    r=None,
):
    """MeanFlow average velocity u(z_t, t, h=t-r)."""
    if r is None:
        r = torch.zeros_like(t)
    batch_size = z.shape[0]
    h = (t - r).view(batch_size)
    u_pred = denoiser(
        z,
        image_latent,
        t.flatten(),
        h_input=h,
        return_velocity=False,
    )
    return u_pred


def _meanflow_step(
    denoiser,
    z,
    image_latent,
    t,
    t_next,
    t_eps,
):
    u_pred = _forward_sample(
        denoiser,
        z,
        image_latent,
        t,
        t_eps,
        r=t_next,
    )
    z_next = z - (t - t_next) * u_pred
    return z_next


def _color_map(num_classes):
    mgu_colors = {index: list(color) for index, color in enumerate(CLASS_COLORS)}
    return {
        index: mgu_colors.get(index, [255, 255, 255]) for index in range(num_classes)
    }


def _normalize_to_uint8(array):
    array = np.asarray(array, dtype=np.float32)
    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return np.zeros(array.shape, dtype=np.uint8)
    finite_values = array[finite_mask]
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    if max_value - min_value < 1e-8:
        return np.zeros(array.shape, dtype=np.uint8)
    array = (array - min_value) / (max_value - min_value)
    array = np.clip(array * 255.0, 0, 255)
    return array.astype(np.uint8)


def _label_to_rgb(label, colors):
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_index, color in colors.items():
        rgb[label == class_index] = color
    return rgb


def _case_png_name(case_name):
    """Use a lossless, directly viewable PNG name for every saved label."""

    stem = os.path.splitext(os.path.basename(str(case_name)))[0]
    return stem + ".png"


def _save_rgb_label(label, colors, output_path):
    """Save a categorical label as an explicitly RGB color image."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(_label_to_rgb(label, colors), mode="RGB").save(
        output_path,
        format="PNG",
    )


def _save_label_id_mask(label, output_path):
    """Keep class ids separately from the visible RGB result."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(np.asarray(label, dtype=np.uint8), mode="L").save(
        output_path,
        format="PNG",
    )


def _save_color_legend(colors, output_dir):
    """Write a color swatch and RGB table beside test predictions."""

    row_height = 32
    width = 360
    canvas = Image.new("RGB", (width, row_height * len(MGU_CLASS_NAMES)), "white")
    draw = ImageDraw.Draw(canvas)
    lines = []
    for class_index, class_name in enumerate(MGU_CLASS_NAMES):
        rgb = tuple(int(value) for value in colors[class_index])
        top = class_index * row_height
        draw.rectangle((0, top, 62, top + row_height - 1), fill=rgb)
        draw.text(
            (72, top + 9), f"{class_index}: {class_name}  RGB{rgb}", fill=(0, 0, 0)
        )
        lines.append(f"{class_index}\t{class_name}\tRGB{rgb}")
    canvas.save(os.path.join(output_dir, "color_legend.png"), format="PNG")
    with open(
        os.path.join(output_dir, "color_legend.txt"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")


def _resize_latent_for_visual(latent, image_height, image_width):
    if latent is None:
        return None
    return F.interpolate(
        latent.detach().cpu(),
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )


def _flow_noised_and_direct_denoised_latent(
    clean_label_latent,
    image_latent,
    denoiser,
    visual_t,
    noise_scale,
):
    """MeanFlow noisy GT latent at a fixed t, then one-step clean prediction."""
    visual_t = float(np.clip(float(visual_t), 0.0, 1.0))
    time = torch.full(
        (clean_label_latent.shape[0],),
        visual_t,
        device=clean_label_latent.device,
        dtype=clean_label_latent.dtype,
    )
    time_view = time.view(-1, *([1] * (clean_label_latent.ndim - 1)))
    noise = torch.randn_like(clean_label_latent) * float(noise_scale)
    noised_label_latent = (1.0 - time_view) * clean_label_latent + time_view * noise
    u = denoiser(
        noised_label_latent,
        image_latent,
        time,
        h_input=time,
        checkpoint_profile="visualization",
        return_velocity=False,
    )
    denoised_from_noised_latent = noised_label_latent - time_view * u
    return noised_label_latent.detach(), denoised_from_noised_latent.detach()


def _append_named_latent_panels(latent_resized, sample_index, prefix, panels):
    if latent_resized is None:
        return
    for channel_index in range(latent_resized.shape[1]):
        channel_panel = _normalize_to_uint8(
            latent_resized[sample_index, channel_index].numpy()
        )
        channel_panel = np.repeat(channel_panel[..., None], 3, axis=2)
        panels.append((f"{prefix}_ch{channel_index:02d}.png", channel_panel))


def _save_denoised_channel_visualizations(
    image,
    labels,
    image_latent,
    clean_label_latent,
    noised_label_latent,
    denoised_from_noised_latent,
    sampled_denoised_latent,
    case_names,
    colors,
    save_dir,
    visual_t,
    layout="joined",
):
    """Save image | GT | image/image-label latent process visualizations."""
    layout = str(layout).lower()
    if layout not in {"joined", "separate"}:
        raise ValueError(
            f"Unknown denoise visualization layout '{layout}'. "
            "Expected 'joined' or 'separate'."
        )
    os.makedirs(save_dir, exist_ok=True)
    readme_path = os.path.join(save_dir, "_panel_order.txt")
    with open(readme_path, "w", encoding="utf-8") as handle:
        if layout == "separate":
            handle.write(
                "Separate layout: every panel/channel is saved as one PNG in a "
                "per-sample folder.\n"
                "Files: 01_input_oct_image.png, 02_ground_truth_label.png, "
                "03_image_latent_chXX.png, 04_clean_label_latent_chXX.png, "
                "05_noised_label_latent_chXX.png, "
                "06_direct_denoised_label_latent_chXX.png, "
                "07_sampled_denoised_label_latent_chXX.png\n\n"
            )
        handle.write(
            "Panel order per sample:\n"
            "1. input OCT image\n"
            "2. GT label\n"
            "3. image latent channels\n"
            "4. clean label latent channels from labelEncoder(GT label condition)\n"
            f"5. noised label latent channels, z_t = (1-t)*x + t*noise, t={float(visual_t):.4f}\n"
            "6. direct denoised label latent channels, z_t - t*u(z_t, image_latent, t)\n"
            "7. random-sampling denoised latent channels after MeanFlow sampling\n"
        )
    image_height, image_width = image.shape[-2:]

    image_latent_resized = _resize_latent_for_visual(
        image_latent,
        image_height,
        image_width,
    )
    clean_resized = _resize_latent_for_visual(
        clean_label_latent,
        image_height,
        image_width,
    )
    noised_resized = _resize_latent_for_visual(
        noised_label_latent,
        image_height,
        image_width,
    )
    direct_denoised_resized = _resize_latent_for_visual(
        denoised_from_noised_latent,
        image_height,
        image_width,
    )
    sampled_denoised_resized = _resize_latent_for_visual(
        sampled_denoised_latent,
        image_height,
        image_width,
    )
    label_resized = (
        F.interpolate(
            labels.unsqueeze(1).float(),
            size=(image_height, image_width),
            mode="nearest",
        )
        .squeeze(1)
        .long()
    )

    image_cpu = image.detach().cpu()
    for sample_index, case_name in enumerate(case_names):
        image_panel = _normalize_to_uint8(image_cpu[sample_index, 0].numpy())
        image_panel = np.repeat(image_panel[..., None], 3, axis=2)

        label_panel = _label_to_rgb(label_resized[sample_index].numpy(), colors)

        named_panels = [
            ("01_input_oct_image.png", image_panel),
            ("02_ground_truth_label.png", label_panel),
        ]
        _append_named_latent_panels(
            image_latent_resized,
            sample_index,
            "03_image_latent",
            named_panels,
        )
        _append_named_latent_panels(
            clean_resized,
            sample_index,
            "04_clean_label_latent",
            named_panels,
        )
        _append_named_latent_panels(
            noised_resized,
            sample_index,
            "05_noised_label_latent",
            named_panels,
        )
        _append_named_latent_panels(
            direct_denoised_resized,
            sample_index,
            "06_direct_denoised_label_latent",
            named_panels,
        )
        _append_named_latent_panels(
            sampled_denoised_resized,
            sample_index,
            "07_sampled_denoised_label_latent",
            named_panels,
        )

        if layout == "separate":
            sample_dir = os.path.join(
                save_dir,
                os.path.splitext(_case_png_name(case_name))[0],
            )
            os.makedirs(sample_dir, exist_ok=True)
            legacy_joined_path = os.path.join(save_dir, str(case_name))
            if os.path.isfile(legacy_joined_path):
                os.remove(legacy_joined_path)
            for filename, panel in named_panels:
                Image.fromarray(panel, mode="RGB").save(
                    os.path.join(sample_dir, filename),
                    format="PNG",
                )
        else:
            canvas = np.concatenate(
                [panel for _, panel in named_panels],
                axis=1,
            )
            output_path = os.path.join(save_dir, case_name)
            os.makedirs(os.path.dirname(output_path) or save_dir, exist_ok=True)
            Image.fromarray(canvas).save(output_path)


@torch.no_grad()
def evaluate_label_vae(
    dataloader,
    label_encoder,
    label_decoder,
    device=None,
    num_classes=11,
    test_save_path=None,
):
    """Test Stage-1 label reconstruction at the original MGU resolution."""
    if int(num_classes) != 11:
        raise ValueError(
            f"MGU Stage-1 testing requires num_classes=11, got {num_classes}"
        )
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_encoder.to(device).eval()
    label_decoder.to(device).eval()
    colors = _color_map(num_classes)
    if test_save_path is not None:
        os.makedirs(test_save_path, exist_ok=True)
        _save_color_legend(colors, test_save_path)
        with open(
            os.path.join(test_save_path, "_label_output_layout.txt"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                "color_predictions/: Stage-1 reconstructed labels (RGB PNG)\n"
                "color_ground_truth/: ground-truth labels (RGB PNG)\n"
                "label_ids/: Stage-1 reconstructed class-id masks "
                "(grayscale PNG; values 0..10)\n"
            )

    all_preds, all_labels, all_case_names = [], [], []
    all_hd95 = []
    all_sample_metrics = []
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in tqdm(dataloader, desc="Stage1 Label VAE Test", ncols=110):
        label_condition = batch["sdf"].to(device)
        labels = batch["label"]
        case_names = batch["case_name"]
        latent = label_encoder(label_condition)
        logits_model = label_decoder(latent, None)
        original_height, original_width = labels.shape[-2:]
        prediction_model = torch.argmax(logits_model, dim=1)
        prediction_original = (
            F.interpolate(
                prediction_model.unsqueeze(1).float(),
                size=(original_height, original_width),
                mode="nearest",
            )
            .squeeze(1)
            .long()
        )
        prediction = prediction_original.cpu().numpy()
        labels_np = labels.numpy()
        all_preds.append(prediction)
        all_labels.append(labels_np)
        all_case_names.extend(case_names)
        confusion_matrix += calculate_confusion_matrix(
            labels_np,
            prediction,
            num_classes=num_classes,
        )

        for sample_index, case_name in enumerate(case_names):
            _, hd95_list, _ = calculate_multiclass_metrics(
                y_true_np=labels_np[sample_index],
                y_pred_np=prediction[sample_index],
                num_classes=num_classes,
                ignore_index=0,
            )
            all_hd95.append(hd95_list)
            all_sample_metrics.append(
                summarize_sample_metrics(
                    case_name=case_name,
                    y_true_np=labels_np[sample_index],
                    y_pred_np=prediction[sample_index],
                    hd95_per_class=hd95_list,
                    num_classes=num_classes,
                )
            )
            if test_save_path is not None:
                png_name = _case_png_name(case_name)
                _save_rgb_label(
                    prediction[sample_index],
                    colors,
                    os.path.join(test_save_path, "color_predictions", png_name),
                )
                _save_rgb_label(
                    labels_np[sample_index],
                    colors,
                    os.path.join(test_save_path, "color_ground_truth", png_name),
                )
                _save_label_id_mask(
                    prediction[sample_index],
                    os.path.join(test_save_path, "label_ids", png_name),
                )

    results = {
        "preds": all_preds,
        "labels": all_labels,
        "case_names": all_case_names,
        "sample_metrics": all_sample_metrics,
    }
    if all_hd95:
        results.update(aggregate_metrics(confusion_matrix, all_hd95))
    return results


@torch.no_grad()
def segment(
    dataloader,
    labelEncoder,
    labelDecoder,
    imageEncoder,
    denoiser,
    model_directory=None,
    samplingSteps=50,
    sampler="meanflow",
    n_eval=1,
    sigma=0,
    device=None,
    plots=False,
    inference=True,
    first_batch_only=False,
    compute_metrics=True,
    num_classes=11,
    test_save_path=None,
    save_denoise_visuals=False,
    denoise_visual_layout="joined",
    noise_scale=1.0,
    t_eps=0.05,
    latent_visual_t=0.5,
):
    """
    Generate a clean latent with MeanFlow sampling, decode label
    logits, then convert logits to MGU categorical labels for metrics and PNG.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if labelEncoder is not None:
        labelEncoder.to(device).eval()
    labelDecoder.to(device).eval()
    imageEncoder.to(device).eval()
    denoiser.to(device).eval()

    if inference and model_directory is not None:
        model_pairs = [
            (labelDecoder, "labelDecoder.pt"),
            (imageEncoder, "imageEncoder.pt"),
            (denoiser, "denoiser.pt"),
        ]
        if labelEncoder is not None:
            model_pairs.insert(0, (labelEncoder, "labelEncoder.pt"))
        for model, filename in model_pairs:
            path = os.path.join(model_directory, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{filename} not found in {model_directory}")
            state_dict = torch.load(path, map_location=device, weights_only=True)
            if filename == "denoiser.pt":
                load_denoiser_state_dict(model, state_dict)
            elif filename == "labelDecoder.pt":
                load_label_decoder_state_dict(model, state_dict, filename)
            else:
                model.load_state_dict(state_dict)
            model.to(device).eval()
            print(f"{filename} loaded.")

    sampling_steps = int(samplingSteps)
    if sampling_steps < 1:
        raise ValueError("samplingSteps must be at least 1")
    method = str(sampler).lower()
    if method != "meanflow":
        raise ValueError(
            f"Unknown sampler '{sampler}'. MeanFlow inference uses sampler='meanflow'."
        )
    evaluation_runs = int(n_eval)
    if evaluation_runs < 1:
        raise ValueError("n_eval must be at least 1")
    metric_source = "segmentation prediction"
    if compute_metrics:
        print(f"Metric prediction source: {metric_source}", flush=True)

    colors = _color_map(num_classes)
    if test_save_path is not None:
        os.makedirs(test_save_path, exist_ok=True)
        _save_color_legend(colors, test_save_path)
        with open(
            os.path.join(test_save_path, "_label_output_layout.txt"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                "color_predictions/: segmentation predictions (RGB PNG)\n"
                "color_ground_truth/: ground-truth labels (RGB PNG)\n"
                "label_ids/: categorical class-id masks (grayscale PNG; values 0..10)\n"
                f"metric_source: {metric_source}\n"
            )
    all_preds, all_labels, all_case_names = [], [], []
    all_hd95 = []
    all_sample_metrics = []
    confusion_matrix = (
        np.zeros((num_classes, num_classes), dtype=np.int64)
        if compute_metrics
        else None
    )

    for batch_idx, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        labels = batch["label"]
        case_names = batch["case_name"]
        if sigma != 0:
            image = add_noise(image, sigma).to(device)

        clean_label_latent = None
        noised_label_latent = None
        denoised_from_noised_latent = None
        if save_denoise_visuals and labelEncoder is not None:
            label_condition = batch["sdf"].to(device)
            clean_label_latent = labelEncoder(label_condition)

        image_latent, image_skip_features = imageEncoder(
            image,
            return_skips=True,
        )
        deepest_image_feature = image_skip_features[-1]
        if clean_label_latent is not None:
            (
                noised_label_latent,
                denoised_from_noised_latent,
            ) = _flow_noised_and_direct_denoised_latent(
                clean_label_latent=clean_label_latent,
                image_latent=image_latent,
                denoiser=denoiser,
                visual_t=latent_visual_t,
                noise_scale=noise_scale,
            )
        sampled_logits = []
        sampled_latents = []
        latent_channels = getattr(denoiser, "label_latent_channels", None)
        if latent_channels is None:
            latent_channels = denoiser.out_conv.out_channels
        latent_channels = int(latent_channels)
        if latent_channels <= 0:
            raise ValueError(
                "Cannot infer sampled label-latent channels from denoiser. "
                f"Got {latent_channels}."
            )
        for evaluation_index in range(evaluation_runs):
            latent = torch.randn(
                image_latent.shape[0],
                latent_channels,
                image_latent.shape[2],
                image_latent.shape[3],
                device=image_latent.device,
                dtype=image_latent.dtype,
            ) * float(noise_scale)
            timesteps = (
                torch.linspace(
                    1.0,
                    0.0,
                    sampling_steps + 1,
                    device=device,
                    dtype=image_latent.dtype,
                )
                .view(-1, *([1] * latent.ndim))
                .expand(
                    -1,
                    latent.shape[0],
                    -1,
                    -1,
                    -1,
                )
            )
            latent_steps = [latent.clone()] if plots and evaluation_index == 0 else None

            for step_index in tqdm(
                range(sampling_steps),
                desc=(
                    f"Sampling batch {batch_idx} "
                    f"[{evaluation_index + 1}/{evaluation_runs}]"
                ),
            ):
                t = timesteps[step_index]
                t_next = timesteps[step_index + 1]
                latent = _meanflow_step(
                    denoiser,
                    latent,
                    image_latent,
                    t,
                    t_next,
                    t_eps,
                )
                if latent_steps is not None:
                    latent_steps.append(latent.clone())

            if latent_steps is not None:
                plot_sampling([item.cpu().numpy() for item in latent_steps])
            sampled_latents.append(latent.detach())
            sampled_logits.append(
                labelDecoder(
                    latent,
                    image_skip_features,
                    image_latent_feature=deepest_image_feature,
                )
            )

        predicted_logits_model = torch.stack(sampled_logits, dim=0).mean(dim=0)
        sampled_latent_model = torch.stack(sampled_latents, dim=0).mean(dim=0)

        original_height, original_width = labels.shape[-2:]
        # Map logits back to the true acquisition resolution before argmax,
        # metric computation, and PNG export.
        interpolated_logits = F.interpolate(
            predicted_logits_model,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )
        predicted_labels = torch.argmax(interpolated_logits, dim=1).long()
        prediction = predicted_labels.cpu().numpy()
        metric_prediction = prediction
        labels_np = labels.numpy()

        all_preds.append(prediction)
        all_labels.append(labels_np)
        all_case_names.extend(case_names)
        if confusion_matrix is not None:
            confusion_matrix += calculate_confusion_matrix(
                labels_np,
                metric_prediction,
                num_classes=num_classes,
            )

        if test_save_path is not None:
            if save_denoise_visuals:
                _save_denoised_channel_visualizations(
                    image=image,
                    labels=labels,
                    image_latent=image_latent,
                    clean_label_latent=clean_label_latent,
                    noised_label_latent=noised_label_latent,
                    denoised_from_noised_latent=denoised_from_noised_latent,
                    sampled_denoised_latent=sampled_latent_model,
                    case_names=case_names,
                    colors=colors,
                    save_dir=os.path.join(test_save_path, "denoise_visualizations"),
                    visual_t=latent_visual_t,
                    layout=denoise_visual_layout,
                )
            for sample_index, case_name in enumerate(case_names):
                pred_mask = prediction[sample_index]
                gt_mask = labels_np[sample_index]
                png_name = _case_png_name(case_name)
                _save_rgb_label(
                    pred_mask,
                    colors,
                    os.path.join(test_save_path, "color_predictions", png_name),
                )
                _save_rgb_label(
                    gt_mask,
                    colors,
                    os.path.join(test_save_path, "color_ground_truth", png_name),
                )
                _save_label_id_mask(
                    pred_mask,
                    os.path.join(test_save_path, "label_ids", png_name),
                )
        if compute_metrics:
            for sample_index, case_name in enumerate(case_names):
                _, hd95_list, _ = calculate_multiclass_metrics(
                    y_true_np=labels_np[sample_index],
                    y_pred_np=metric_prediction[sample_index],
                    num_classes=num_classes,
                    ignore_index=0,
                )
                all_hd95.append(hd95_list)
                all_sample_metrics.append(
                    summarize_sample_metrics(
                        case_name=case_name,
                        y_true_np=labels_np[sample_index],
                        y_pred_np=metric_prediction[sample_index],
                        hd95_per_class=hd95_list,
                        num_classes=num_classes,
                    )
                )

        if first_batch_only:
            break

    results = {
        "preds": all_preds,
        "labels": all_labels,
        "case_names": all_case_names,
        "sample_metrics": all_sample_metrics,
    }
    if compute_metrics and all_hd95:
        results.update(aggregate_metrics(confusion_matrix, all_hd95))
    return results
