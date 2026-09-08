"""LSFSeg training and evaluation entry point."""

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from openpyxl import Workbook
from torch.utils.data import DataLoader

from network import data, inference, models, training
from network.config import CLASS_NAMES, SELECTED_STAGE1_JSON, SELECTED_STAGE1_TXT

MODEL_NAME = "LSFSeg"


def resolve_project_path(path):
    """Keep configured paths relative to the LSFSeg working directory."""
    return os.path.normpath(path)


def build_parser():
    parser = argparse.ArgumentParser(
        description=f"{MODEL_NAME} latent MeanFlow segmentation"
    )
    parser.add_argument("--data_path", default=os.path.join("Data", "MGU"))
    parser.add_argument("--dataset", default="MGU", choices=["MGU"])
    parser.add_argument("--list_dir", default=os.path.join("Data", "MGU", "lists"))
    parser.add_argument("--num_classes", type=int, choices=[11], default=11)
    parser.add_argument(
        "--output_dir",
        default=os.path.join("output", "MGU", "single_split", "stage2"),
    )
    parser.add_argument("--excel_name", default="batch_test_results.xlsx")
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument(
        "--stage2_additional_epochs",
        type=int,
        default=None,
        help="number of epochs to train after loading the selected Stage-2 checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=0.002)
    parser.add_argument(
        "--stage2_lr_decay_epochs",
        type=int,
        default=0,
        help="halve/multiply the Stage-2 learning rate after this many resumed epochs; 0 disables",
    )
    parser.add_argument(
        "--stage2_lr_decay_gamma",
        type=float,
        default=0.5,
        help="Stage-2 learning-rate multiplier applied at every decay interval",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        choices=[512],
        default=[512, 512],
        metavar=("H", "W"),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--pth_path", default=None)
    parser.add_argument(
        "--is_savenii",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save categorical prediction PNGs",
    )
    parser.add_argument(
        "--save_denoise_visuals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "save image | GT label | image latent channels | clean label "
            "latent channels | noised label latent channels | direct-denoised "
            "label latent channels | random-sampling denoised latent channels"
        ),
    )
    parser.add_argument(
        "--denoise_visual_layout",
        choices=("joined", "separate"),
        default="joined",
        help=(
            "joined saves one concatenated denoise-process PNG per sample; "
            "separate saves every panel/channel as its own PNG"
        ),
    )
    parser.add_argument(
        "--latent_visual_t",
        type=float,
        default=0.5,
        help=(
            "fixed flow time t used only for visualizing noised test label latent: "
            "z_t = (1 - t) * clean_latent + t * noise"
        ),
    )
    parser.add_argument(
        "--stage2_resume_dir",
        default=None,
        help="directory containing tagged Stage-2 checkpoint sets used to warm-start training",
    )
    parser.add_argument(
        "--stage2_resume_tag",
        default="latest",
        help="Stage-2 checkpoint tag to load; 'latest' selects the largest complete epoch_N set",
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test_all", action="store_true")
    parser.add_argument("--weight_tag", default="best_epoch")
    parser.add_argument(
        "--min_test_epoch",
        type=int,
        default=0,
        help=(
            "when --test_all is enabled, only evaluate epoch_N checkpoints "
            "with N >= this value"
        ),
    )
    parser.add_argument(
        "--max_test_epoch",
        type=int,
        default=None,
        help="when --test_all is enabled, only evaluate epoch_N checkpoints with N <= this value",
    )
    parser.add_argument(
        "--include_best_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "include best_epoch when discovering checkpoints; multi-epoch "
            "--test_all always includes best_epoch when the checkpoint exists"
        ),
    )

    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=1,
        help="MeanFlow inference steps; default 1 is one-step denoising",
    )
    parser.add_argument(
        "--sampler",
        default="meanflow",
        choices=["meanflow"],
        help="MeanFlow inference sampler; Heun/Euler branches are not used",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=1,
        help="number of independently sampled label-logit predictions averaged before argmax",
    )
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--p_mean", type=float, default=0.8)
    parser.add_argument("--p_std", type=float, default=0.8)
    parser.add_argument("--t_eps", type=float, default=0.05)
    parser.add_argument(
        "--meanflow_data_proportion",
        type=float,
        default=0.5,
        help="probability of r=t flow-matching samples in MeanFlow stage-2 training",
    )
    parser.add_argument(
        "--meanflow_norm_p",
        type=float,
        default=1.0,
        help="adaptive MeanFlow loss exponent",
    )
    parser.add_argument(
        "--meanflow_norm_eps",
        type=float,
        default=0.01,
        help="adaptive MeanFlow loss epsilon",
    )
    parser.add_argument("--dual_view_consistency_weight", type=float, default=0.05)
    parser.add_argument("--dual_view_latent_weight", type=float, default=0.25)
    parser.add_argument("--dual_view_warmup_epochs", type=int, default=10)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument(
        "--segmentation_weight",
        type=float,
        default=1.0,
        help="weight multiplying the complete Stage-2 Focal + 2*Dice loss",
    )
    parser.add_argument(
        "--denoise_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--label_condition_mode",
        default="one_hot",
        choices=["one_hot"],
        help="MGU label encoder input: one channel per class",
    )
    parser.add_argument(
        "--train_stage",
        default="flow",
        choices=["flow", "label_vae"],
        help="label_vae trains only the label encoder/decoder; flow runs the original denoising stage",
    )
    parser.add_argument(
        "--latent_fusion_hidden_channels",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--label_vae_dir",
        default=None,
        help="directory containing labelEncoder/labelDecoder checkpoints from --train_stage label_vae",
    )
    parser.add_argument("--stage1_ce_weight", type=float, default=1.0)
    parser.add_argument("--stage1_dice_weight", type=float, default=1.0)
    parser.add_argument("--stage1_kl_weight", type=float, default=1e-6)
    parser.add_argument("--stage1_warmup_steps", type=int, default=10000)
    parser.add_argument(
        "--stage1_lr_scheduler",
        choices=["constant", "warmup_linear"],
        default="warmup_linear",
    )
    parser.add_argument("--stage1_min_lr_scale", type=float, default=0.0)
    parser.add_argument("--stage1_grad_clip", type=float, default=1.0)
    parser.add_argument("--stage1_max_steps", type=int, default=None)
    parser.add_argument("--stage1_save_every", type=int, default=0)
    parser.add_argument("--stage1_save_interval_steps", type=int, default=0)

    parser.add_argument("--deterministic", type=int, default=1)
    return parser


def use_training_mode(parsed_args):
    if parsed_args.train:
        return True
    if parsed_args.eval or parsed_args.test_all:
        return False
    return True


def setup_runtime(parsed_args):
    cudnn.benchmark = not bool(parsed_args.deterministic)
    cudnn.deterministic = bool(parsed_args.deterministic)
    random.seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)
    torch.manual_seed(parsed_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(parsed_args.seed)

    if parsed_args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )
    if parsed_args.device == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("CUDA is not available, running on CPU.", flush=True)
    print(f"Using device: {device}", flush=True)
    return device


def make_dataset(parsed_args, split, training=False):
    target_size = tuple(parsed_args.img_size)
    transform = None
    if training:
        transform = data.RandomGenerator(
            target_size,
            num_classes=parsed_args.num_classes,
            label_condition_mode=parsed_args.label_condition_mode,
            enable_dual_view=parsed_args.train_stage == "flow",
        )
    return data.MGUDataset(
        parsed_args.data_path,
        parsed_args.list_dir,
        split=split,
        transform=transform,
        target_size=target_size,
        num_classes=parsed_args.num_classes,
        label_condition_mode=parsed_args.label_condition_mode,
    )


def make_dataloaders(parsed_args):
    if use_training_mode(parsed_args):
        return (
            make_dataset(parsed_args, "train", training=True),
            make_dataset(parsed_args, "val"),
        )
    test_dataset = make_dataset(parsed_args, "test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=parsed_args.batch_size,
        shuffle=False,
        num_workers=0 if parsed_args.train_stage == "label_vae" else 4,
    )
    return test_dataset, test_loader


def load_components(sample_batch, parsed_args, device):
    return models.load_model(
        sample_batch,
        device=device,
        num_classes=parsed_args.num_classes,
        decoder_skip_mode="concat" if parsed_args.train_stage == "flow" else "none",
        enable_latent_image_fusion=parsed_args.train_stage == "flow",
        latent_fusion_hidden_channels=parsed_args.latent_fusion_hidden_channels,
    )


def checkpoint_path(weight_dir, prefix, tag):
    return os.path.join(weight_dir, f"{prefix}_{tag}.pt")


def _stage1_tag_order_for_tie_break(tag):
    if tag == "best_epoch":
        return -1
    if tag == "last_epoch":
        return 10**9
    for prefix, offset in (("epoch_", 0), ("step_", 10**7)):
        if tag.startswith(prefix):
            try:
                return offset + int(tag.split("_", 1)[1])
            except ValueError:
                return 10**8
    return 10**8


def save_selected_stage1_checkpoint(all_results, weight_dir, excel_output_path):
    """Select the highest foreground Dice checkpoint after full Stage-1 testing."""
    failed_tags = [tag for tag, result in all_results.items() if result is None]
    if failed_tags:
        raise RuntimeError(
            "Stage-1 checkpoint selection requires every discovered checkpoint to test "
            f"successfully; failed={failed_tags}"
        )
    valid_results = {
        tag: result
        for tag, result in all_results.items()
        if result is not None and "mean_dice_all_10" in result
    }
    if not valid_results:
        raise RuntimeError(
            "No valid Stage-1 result was produced; cannot select a checkpoint"
        )
    selected_tag, selected_result = max(
        valid_results.items(),
        key=lambda item: (
            float(item[1]["mean_dice_all_10"]),
            -_stage1_tag_order_for_tie_break(item[0]),
        ),
    )
    payload = {
        "selected_tag": selected_tag,
        "selection_metric": "mean_dice_all_10",
        "selection_split": "test",
        "selection_scope": "all_complete_stage1_checkpoints",
        "mean_dice_all_10": float(selected_result["mean_dice_all_10"]),
        "mean_dice_retina_9_no_optic_disc": float(
            selected_result["mean_dice_retina_9_no_optic_disc"]
        ),
        "mean_iou_all_10": float(selected_result["mean_iou_all_10"]),
        "mean_hd95_all_10": float(selected_result["mean_hd95_all_10"]),
        "excel_path": os.path.abspath(excel_output_path),
        "encoder_path": os.path.abspath(
            checkpoint_path(weight_dir, "labelEncoder", selected_tag)
        ),
        "decoder_path": os.path.abspath(
            checkpoint_path(weight_dir, "labelDecoder", selected_tag)
        ),
    }
    json_path = os.path.join(weight_dir, SELECTED_STAGE1_JSON)
    txt_path = os.path.join(weight_dir, SELECTED_STAGE1_TXT)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(selected_tag + "\n")
    print(
        f"Selected Stage-1 checkpoint: {selected_tag} "
        f"(foreground Dice All_10={selected_result['mean_dice_all_10']:.6f})",
        flush=True,
    )
    print(f"Stage-1 selection metadata saved to: {json_path}", flush=True)
    return payload


def resolve_selected_stage1_checkpoint(vae_dir):
    """Require the Dice-based Stage-1 selection produced by the full test."""
    vae_dir = vae_dir if os.path.isabs(vae_dir) else resolve_project_path(vae_dir)
    json_path = os.path.join(vae_dir, SELECTED_STAGE1_JSON)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Missing Stage-1 Dice selection metadata: {json_path}. "
            "Run mgu_stage1.py --mode test --test_all before Stage-2 training."
        )
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("selection_metric") != "mean_dice_all_10":
        raise ValueError(
            f"Invalid Stage-1 selection metric in {json_path}: "
            f"{payload.get('selection_metric')!r}"
        )
    if payload.get("selection_scope") != "all_complete_stage1_checkpoints":
        raise ValueError(
            f"Stage-1 selection was not produced from all complete checkpoints: {json_path}"
        )
    selected_tag = str(payload.get("selected_tag") or "").strip()
    if not selected_tag:
        raise ValueError(f"Stage-1 selection has no selected_tag: {json_path}")
    required = [
        checkpoint_path(vae_dir, "labelEncoder", selected_tag),
        checkpoint_path(vae_dir, "labelDecoder", selected_tag),
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"Selected Stage-1 checkpoint '{selected_tag}' is incomplete: {missing}"
        )
    print(
        f"Stage-2 will load Dice-selected Stage-1 checkpoint '{selected_tag}' "
        f"from {json_path}",
        flush=True,
    )
    return selected_tag


def load_checkpoint_tag(
    label_encoder,
    image_encoder,
    denoiser,
    label_decoder,
    weight_dir,
    tag,
    device,
    preserve_archived_output=True,
):
    models.load_label_encoder_state_dict(
        label_encoder,
        torch.load(
            checkpoint_path(weight_dir, "labelEncoder", tag),
            map_location=device,
            weights_only=True,
        ),
        checkpoint_name=f"labelEncoder_{tag}.pt",
    )
    models.load_image_encoder_state_dict(
        image_encoder,
        torch.load(
            checkpoint_path(weight_dir, "imageEncoder", tag),
            map_location=device,
            weights_only=True,
        ),
    )
    models.load_denoiser_state_dict(
        denoiser,
        torch.load(
            checkpoint_path(weight_dir, "denoiser", tag),
            map_location=device,
            weights_only=True,
        ),
        preserve_archived_output=preserve_archived_output,
    )
    decoder_state = torch.load(
        checkpoint_path(weight_dir, "labelDecoder", tag),
        map_location=device,
        weights_only=True,
    )
    models.load_label_decoder_state_dict(
        label_decoder,
        decoder_state,
        checkpoint_name=f"labelDecoder_{tag}.pt",
    )
    for model in [label_encoder, image_encoder, denoiser, label_decoder]:
        model.to(device).eval()


def load_label_vae_checkpoint(
    label_encoder,
    label_decoder,
    vae_dir,
    tag,
    device,
    num_classes,
    image_size,
    label_condition_mode,
    load_decoder=True,
):
    vae_dir = vae_dir if os.path.isabs(vae_dir) else resolve_project_path(vae_dir)
    encoder_path = checkpoint_path(vae_dir, "labelEncoder", tag)
    decoder_path = checkpoint_path(vae_dir, "labelDecoder", tag)
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Missing label VAE encoder checkpoint: {encoder_path}")
    if load_decoder and not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Missing label VAE decoder checkpoint: {decoder_path}")
    contract_path = models.validate_label_encoder_contract(
        vae_dir,
        label_encoder,
        num_classes=num_classes,
        image_size=image_size,
        label_condition_mode=label_condition_mode,
    )
    models.load_label_encoder_state_dict(
        label_encoder,
        torch.load(encoder_path, map_location=device, weights_only=True),
        checkpoint_name=encoder_path,
    )
    print(f"Validated Stage-1/Stage-2 encoder contract: {contract_path}", flush=True)
    if load_decoder:
        decoder_state = torch.load(decoder_path, map_location=device, weights_only=True)
        label_decoder.load_state_dict(decoder_state, strict=True)
        print(
            f"Loaded label VAE encoder/decoder checkpoint '{tag}' from {vae_dir}",
            flush=True,
        )
    else:
        print(
            f"Loaded label VAE encoder checkpoint '{tag}' from {vae_dir}; "
            "decoder is initialized with the current concat-skip architecture.",
            flush=True,
        )


def discover_checkpoint_tags(
    weight_dir, min_epoch=0, max_epoch=None, include_best_checkpoint=True
):
    tags = []
    min_epoch = int(min_epoch)
    for image_path in glob.glob(os.path.join(weight_dir, "imageEncoder_*.pt")):
        tag = os.path.basename(image_path)[len("imageEncoder_") : -len(".pt")]
        if tag == "best_epoch" and not include_best_checkpoint:
            continue
        if tag.startswith("epoch_"):
            try:
                epoch_index = int(tag.split("_", 1)[1])
            except ValueError:
                continue
            if epoch_index < min_epoch:
                continue
            if max_epoch is not None and epoch_index > int(max_epoch):
                continue
        elif min_epoch > 0:
            continue
        required = [
            checkpoint_path(weight_dir, "labelEncoder", tag),
            checkpoint_path(weight_dir, "imageEncoder", tag),
            checkpoint_path(weight_dir, "denoiser", tag),
            checkpoint_path(weight_dir, "labelDecoder", tag),
        ]
        if all(os.path.exists(path) for path in required):
            tags.append(tag)

    def sort_key(tag):
        if tag == "best_epoch":
            return (-1, 0)
        if tag.startswith("epoch_"):
            try:
                return (0, int(tag.split("_", 1)[1]))
            except ValueError:
                return (0, tag)
        return (1, tag)

    return sorted(set(tags), key=sort_key)


def resolve_stage2_resume_checkpoint(weight_dir, requested_tag="latest"):
    """Resolve a complete tagged Stage-2 checkpoint and its next epoch number."""

    def checkpoint_epoch(tag):
        for prefix in ("epoch_", "best_epoch_"):
            if tag.startswith(prefix):
                suffix = tag[len(prefix) :].split("_run", 1)[0]
                try:
                    return int(suffix)
                except ValueError:
                    return None
        return None

    weight_dir = (
        weight_dir if os.path.isabs(weight_dir) else resolve_project_path(weight_dir)
    )
    if not os.path.isdir(weight_dir):
        raise FileNotFoundError(
            f"Stage-2 resume directory does not exist: {weight_dir}"
        )

    tags = discover_checkpoint_tags(weight_dir, include_best_checkpoint=True)
    if not tags:
        raise FileNotFoundError(
            f"No complete Stage-2 checkpoint sets found in resume directory: {weight_dir}"
        )

    requested_tag = str(requested_tag or "latest").strip()
    if requested_tag.lower() == "latest":
        numbered_tags = []
        for tag in tags:
            epoch_number = checkpoint_epoch(tag)
            if epoch_number is not None:
                numbered_tags.append((epoch_number, tag))
        if numbered_tags:
            epoch_number, selected_tag = max(numbered_tags)
            return weight_dir, selected_tag, epoch_number + 1
        if "best_epoch" in tags:
            return weight_dir, "best_epoch", 1
        return weight_dir, tags[-1], 1

    if requested_tag not in tags:
        raise FileNotFoundError(
            f"Requested Stage-2 resume tag '{requested_tag}' is not a complete checkpoint "
            f"set in {weight_dir}; available={tags}"
        )
    epoch_number = checkpoint_epoch(requested_tag)
    if epoch_number is not None:
        return weight_dir, requested_tag, epoch_number + 1
    return weight_dir, requested_tag, 1


def validate_stage2_encoder_contract_if_present(
    weight_dir,
    label_encoder,
    num_classes,
    image_size,
    label_condition_mode,
):
    """Validate optional metadata without making legacy Stage-2 weights unusable.

    The checkpoint loader still performs strict key and tensor-shape validation
    for every LabelEncoder state dict. This fallback only applies when an older
    or copied weight directory lacks ``label_encoder_contract.json``.
    """

    contract_path = os.path.join(
        weight_dir,
        models.LABEL_ENCODER_CONTRACT_FILENAME,
    )
    if not os.path.isfile(contract_path):
        print(
            "WARNING: Stage-2 label encoder contract is missing: "
            f"{contract_path}. Continuing with strict LabelEncoder checkpoint "
            "key/shape validation; no checkpoint file will be modified.",
            flush=True,
        )
        return None
    return models.validate_label_encoder_contract(
        weight_dir,
        label_encoder,
        num_classes=num_classes,
        image_size=image_size,
        label_condition_mode=label_condition_mode,
    )


def verify_stage2_resume_encoder_matches_stage1(label_encoder, weight_dir, tag, device):
    """Prevent a warm start from silently changing the frozen Stage-1 latent space."""
    resume_path = checkpoint_path(weight_dir, "labelEncoder", tag)
    resume_state = torch.load(resume_path, map_location=device, weights_only=True)
    stage1_state = label_encoder.state_dict()
    if set(resume_state) != set(stage1_state):
        missing = sorted(set(stage1_state) - set(resume_state))
        unexpected = sorted(set(resume_state) - set(stage1_state))
        raise RuntimeError(
            "Stage-2 resume label encoder does not match the selected Stage-1 encoder: "
            f"missing={missing}, unexpected={unexpected}"
        )
    mismatches = [
        key
        for key in stage1_state
        if resume_state[key].shape != stage1_state[key].shape
        or not torch.equal(resume_state[key], stage1_state[key])
    ]
    if mismatches:
        raise RuntimeError(
            "Stage-2 resume label encoder weights differ from the Dice-selected "
            f"Stage-1 encoder; mismatched keys={mismatches}"
        )
    print(
        f"Verified Stage-2 resume encoder matches selected Stage-1 encoder: {resume_path}",
        flush=True,
    )


def discover_label_vae_checkpoint_tags(weight_dir, min_epoch=0, max_epoch=None):
    """Discover complete Stage-1 encoder/decoder checkpoint pairs."""
    tags = []
    min_epoch = int(min_epoch)
    for encoder_path in glob.glob(os.path.join(weight_dir, "labelEncoder_*.pt")):
        tag = os.path.basename(encoder_path)[len("labelEncoder_") : -len(".pt")]
        if tag.startswith("epoch_"):
            try:
                epoch_index = int(tag.split("_", 1)[1])
            except ValueError:
                continue
            if epoch_index < min_epoch:
                continue
            if max_epoch is not None and epoch_index > int(max_epoch):
                continue
        decoder_path = checkpoint_path(weight_dir, "labelDecoder", tag)
        if os.path.exists(decoder_path):
            tags.append(tag)

    def sort_key(tag):
        if tag == "best_epoch":
            return (-1, 0)
        if tag.startswith("epoch_"):
            try:
                return (0, int(tag.split("_", 1)[1]))
            except ValueError:
                return (0, tag)
        if tag.startswith("step_"):
            try:
                return (1, int(tag.split("_", 1)[1]))
            except ValueError:
                return (1, tag)
        return (2, tag)

    return sorted(set(tags), key=sort_key)


def run_segment(
    parsed_args,
    dataloader,
    label_encoder,
    label_decoder,
    image_encoder,
    denoiser,
    device,
    test_save_path=None,
):
    return inference.segment(
        dataloader=dataloader,
        labelEncoder=label_encoder,
        labelDecoder=label_decoder,
        imageEncoder=image_encoder,
        denoiser=denoiser,
        model_directory=parsed_args.output_dir,
        samplingSteps=parsed_args.sampling_steps,
        sampler=parsed_args.sampler,
        n_eval=parsed_args.n_eval,
        sigma=parsed_args.sigma,
        plots=False,
        device=device,
        inference=False,
        first_batch_only=False,
        compute_metrics=True,
        num_classes=parsed_args.num_classes,
        test_save_path=test_save_path,
        save_denoise_visuals=parsed_args.save_denoise_visuals,
        denoise_visual_layout=getattr(parsed_args, "denoise_visual_layout", "joined"),
        noise_scale=parsed_args.noise_scale,
        t_eps=parsed_args.t_eps,
        latent_visual_t=parsed_args.latent_visual_t,
    )


def _append_sample_metric_row(worksheet, tag, metric):
    worksheet.append(
        [
            tag,
            metric.get("case_name"),
            metric.get("mean_dice_all_10"),
            metric.get("mean_iou_all_10"),
            metric.get("mean_hd95_all_10"),
            metric.get("false_positive_pixels"),
            metric.get("false_negative_pixels"),
            metric.get("foreground_gt_pixels"),
            metric.get("foreground_pred_pixels"),
        ]
    )


def _sorted_sample_metrics(sample_metrics, key, reverse):
    return sorted(
        sample_metrics,
        key=lambda item: float(item.get(key, 0.0)),
        reverse=reverse,
    )[:10]


def save_results_to_excel(all_results, output_path, num_classes):
    if int(num_classes) != 11:
        raise ValueError(f"MGU requires num_classes=11, got {num_classes}")
    class_names = list(CLASS_NAMES[1:])

    workbook = Workbook()
    ws_dice = workbook.active
    ws_dice.title = "DICE"
    ws_hd95 = workbook.create_sheet("HD95")
    ws_iou = workbook.create_sheet("IOU")
    ws_sample = workbook.create_sheet("Sample_Metrics")
    ws_rank = workbook.create_sheet("Sample_Rankings")
    ws_dice.append(
        ["PTH_Name"]
        + class_names
        + ["Mean_DICE_All_10", "Mean_DICE_Retina_9_No_OpticDisc"]
    )
    ws_hd95.append(
        ["PTH_Name"]
        + class_names
        + ["Mean_HD95_All_10", "Mean_HD95_Retina_9_No_OpticDisc"]
    )
    ws_iou.append(
        ["PTH_Name"]
        + class_names
        + ["Mean_IOU_All_10", "Mean_IOU_Retina_9_No_OpticDisc"]
    )
    sample_header = [
        "PTH_Name",
        "Case_Name",
        "Dice_All_10",
        "IoU_All_10",
        "HD95_All_10",
        "False_Positive_Pixels",
        "False_Negative_Pixels",
        "Foreground_GT_Pixels",
        "Foreground_Pred_Pixels",
    ]
    ws_sample.append(sample_header)
    ws_rank.append(["Ranking_Type", "Rank"] + sample_header)

    for tag, result in all_results.items():
        if result is None:
            error_row = [tag] + ["Error"] * (len(class_names) + 2)
            ws_dice.append(error_row)
            ws_hd95.append(error_row)
            ws_iou.append(error_row)
            continue
        for metric_key in ("dice_per_class", "hd95_per_class", "iou_per_class"):
            if len(result[metric_key]) != len(class_names):
                raise ValueError(
                    f"{tag} returned {len(result[metric_key])} values for {metric_key}; "
                    f"MGU requires {len(class_names)} foreground-class values"
                )
        ws_dice.append(
            [tag]
            + result["dice_per_class"]
            + [
                result["mean_dice_all_10"],
                result["mean_dice_retina_9_no_optic_disc"],
            ]
        )
        ws_hd95.append(
            [tag]
            + result["hd95_per_class"]
            + [
                result["mean_hd95_all_10"],
                result["mean_hd95_retina_9_no_optic_disc"],
            ]
        )
        ws_iou.append(
            [tag]
            + result["iou_per_class"]
            + [
                result["mean_iou_all_10"],
                result["mean_iou_retina_9_no_optic_disc"],
            ]
        )
        sample_metrics = list(result.get("sample_metrics") or [])
        for metric in sample_metrics:
            _append_sample_metric_row(ws_sample, tag, metric)
        ranking_specs = [
            ("Best_DICE_Top10", "mean_dice_all_10", True),
            ("Worst_DICE_Top10", "mean_dice_all_10", False),
            ("Best_IOU_Top10", "mean_iou_all_10", True),
            ("Worst_IOU_Top10", "mean_iou_all_10", False),
            ("Best_HD95_Top10", "mean_hd95_all_10", False),
            ("Worst_HD95_Top10", "mean_hd95_all_10", True),
            ("Largest_FP_Top10", "false_positive_pixels", True),
            ("Largest_FN_Top10", "false_negative_pixels", True),
        ]
        for ranking_name, key, reverse in ranking_specs:
            for rank_index, metric in enumerate(
                _sorted_sample_metrics(sample_metrics, key, reverse),
                start=1,
            ):
                ws_rank.append([ranking_name, rank_index])
                row = ws_rank.max_row
                values = [
                    tag,
                    metric.get("case_name"),
                    metric.get("mean_dice_all_10"),
                    metric.get("mean_iou_all_10"),
                    metric.get("mean_hd95_all_10"),
                    metric.get("false_positive_pixels"),
                    metric.get("false_negative_pixels"),
                    metric.get("foreground_gt_pixels"),
                    metric.get("foreground_pred_pixels"),
                ]
                for column_index, value in enumerate(values, start=3):
                    ws_rank.cell(row=row, column=column_index, value=value)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    workbook.save(output_path)


def train(parsed_args, device):
    train_dataset, val_dataset = make_dataloaders(parsed_args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=parsed_args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    sample_batch = next(iter(train_loader))
    label_encoder, label_decoder, image_encoder, denoiser = load_components(
        sample_batch,
        parsed_args,
        device,
    )
    if parsed_args.train_stage == "label_vae":
        contract_path = models.save_label_encoder_contract(
            parsed_args.output_dir,
            label_encoder,
            num_classes=parsed_args.num_classes,
            image_size=parsed_args.img_size,
            label_condition_mode=parsed_args.label_condition_mode,
        )
        print(f"Saved Stage-1 encoder contract: {contract_path}", flush=True)
        training.train_label_vae(
            labelEncoder=label_encoder,
            labelDecoder=label_decoder,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=parsed_args.max_epochs,
            batch_size=parsed_args.batch_size,
            lr=parsed_args.base_lr,
            modelDirectory=parsed_args.output_dir,
            device=device,
            num_classes=parsed_args.num_classes,
            ce_weight=parsed_args.stage1_ce_weight,
            dice_weight=parsed_args.stage1_dice_weight,
            kl_weight=parsed_args.stage1_kl_weight,
            warmup_steps=parsed_args.stage1_warmup_steps,
            lr_scheduler=parsed_args.stage1_lr_scheduler,
            min_lr_scale=parsed_args.stage1_min_lr_scale,
            grad_clip=parsed_args.stage1_grad_clip,
            max_steps=parsed_args.stage1_max_steps,
            save_every=parsed_args.stage1_save_every,
            save_interval_steps=parsed_args.stage1_save_interval_steps,
        )
        return

    if not parsed_args.label_vae_dir:
        raise ValueError(
            "MGU Stage-2 training requires --label_vae_dir so its label encoder "
            "can be initialized from the compatible Stage-1 checkpoint."
        )
    selected_stage1_tag = resolve_selected_stage1_checkpoint(parsed_args.label_vae_dir)
    if parsed_args.weight_tag != selected_stage1_tag:
        raise ValueError(
            "MGU Stage-2 must load the Stage-1 label encoder selected by foreground Dice: "
            f"expected weight_tag='{selected_stage1_tag}', got '{parsed_args.weight_tag}'"
        )
    load_label_vae_checkpoint(
        label_encoder,
        label_decoder,
        parsed_args.label_vae_dir,
        parsed_args.weight_tag,
        device,
        num_classes=parsed_args.num_classes,
        image_size=parsed_args.img_size,
        label_condition_mode=parsed_args.label_condition_mode,
        load_decoder=False,
    )
    stage2_start_epoch = 1
    if parsed_args.stage2_resume_dir:
        resume_dir, resume_tag, stage2_start_epoch = resolve_stage2_resume_checkpoint(
            parsed_args.stage2_resume_dir,
            parsed_args.stage2_resume_tag,
        )
        resume_contract = validate_stage2_encoder_contract_if_present(
            resume_dir,
            label_encoder,
            num_classes=parsed_args.num_classes,
            image_size=parsed_args.img_size,
            label_condition_mode=parsed_args.label_condition_mode,
        )
        verify_stage2_resume_encoder_matches_stage1(
            label_encoder,
            resume_dir,
            resume_tag,
            device,
        )
        load_checkpoint_tag(
            label_encoder,
            image_encoder,
            denoiser,
            label_decoder,
            resume_dir,
            resume_tag,
            device,
            preserve_archived_output=False,
        )
        print(
            f"Warm-started Stage-2 from tag '{resume_tag}' in {resume_dir}; "
            f"validated contract={resume_contract}; training starts at epoch "
            f"{stage2_start_epoch} (optimizer state is initialized fresh).",
            flush=True,
        )
    if parsed_args.stage2_additional_epochs is not None:
        additional_epochs = int(parsed_args.stage2_additional_epochs)
        if additional_epochs < 1:
            raise ValueError("--stage2_additional_epochs must be at least 1")
        stage2_end_epoch = stage2_start_epoch + additional_epochs - 1
    else:
        stage2_end_epoch = int(parsed_args.max_epochs)
    print(
        "Stage-2 training range: "
        f"epoch {stage2_start_epoch} through {stage2_end_epoch} "
        f"({stage2_end_epoch - stage2_start_epoch + 1} epoch(s)); "
        f"base_lr={parsed_args.base_lr}, "
        f"lr_decay_every={parsed_args.stage2_lr_decay_epochs} epoch(s), "
        f"lr_decay_gamma={parsed_args.stage2_lr_decay_gamma}",
        flush=True,
    )
    stage2_contract_path = models.save_label_encoder_contract(
        parsed_args.output_dir,
        label_encoder,
        num_classes=parsed_args.num_classes,
        image_size=parsed_args.img_size,
        label_condition_mode=parsed_args.label_condition_mode,
    )
    print(
        f"Saved validated Stage-2 encoder contract: {stage2_contract_path}", flush=True
    )
    print(
        "Stage-2 dual-view consistency: "
        f"weight={parsed_args.dual_view_consistency_weight}, "
        f"latent_weight={parsed_args.dual_view_latent_weight}, "
        f"warmup_epochs={parsed_args.dual_view_warmup_epochs}",
        flush=True,
    )
    print(
        "Stage-2 segmentation loss: "
        f"{parsed_args.segmentation_weight} * "
        "(Focal(alpha=0.25, gamma=2.0) + 2 * Dice)",
        flush=True,
    )
    training.train_models(
        labelEncoder=label_encoder,
        labelDecoder=label_decoder,
        imageEncoder=image_encoder,
        denoiser=denoiser,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=stage2_end_epoch,
        start_epoch=stage2_start_epoch,
        batch_size=parsed_args.batch_size,
        lr=parsed_args.base_lr,
        lr_decay_epochs=parsed_args.stage2_lr_decay_epochs,
        lr_decay_gamma=parsed_args.stage2_lr_decay_gamma,
        modelDirectory=parsed_args.output_dir,
        device=device,
        num_classes=parsed_args.num_classes,
        p_mean=parsed_args.p_mean,
        p_std=parsed_args.p_std,
        noise_scale=parsed_args.noise_scale,
        t_eps=parsed_args.t_eps,
        ema_decay=parsed_args.ema_decay,
        segmentation_weight=parsed_args.segmentation_weight,
        denoise_weight=parsed_args.denoise_weight,
        meanflow_data_proportion=parsed_args.meanflow_data_proportion,
        meanflow_norm_p=parsed_args.meanflow_norm_p,
        meanflow_norm_eps=parsed_args.meanflow_norm_eps,
        dual_view_consistency_weight=parsed_args.dual_view_consistency_weight,
        dual_view_latent_weight=parsed_args.dual_view_latent_weight,
        dual_view_warmup_epochs=parsed_args.dual_view_warmup_epochs,
    )


def evaluate_stage1_label_vae(parsed_args, device):
    """Load and test Stage-1 label reconstruction checkpoints."""
    _, dataloader = make_dataloaders(parsed_args)
    sample_batch = next(iter(dataloader))
    label_encoder, label_decoder = models.load_label_vae_model(
        sample_batch,
        device=device,
        num_classes=parsed_args.num_classes,
    )
    weight_dir = parsed_args.pth_path or parsed_args.output_dir
    weight_dir = (
        weight_dir if os.path.isabs(weight_dir) else resolve_project_path(weight_dir)
    )
    contract_path = models.validate_label_encoder_contract(
        weight_dir,
        label_encoder,
        num_classes=parsed_args.num_classes,
        image_size=parsed_args.img_size,
        label_condition_mode=parsed_args.label_condition_mode,
    )
    print(f"Validated Stage-1 test encoder contract: {contract_path}", flush=True)
    if parsed_args.test_all:
        tags = discover_label_vae_checkpoint_tags(
            weight_dir,
            min_epoch=parsed_args.min_test_epoch,
            max_epoch=parsed_args.max_test_epoch,
        )
        if not tags:
            raise FileNotFoundError(
                f"No Stage-1 encoder/decoder checkpoint pairs in {weight_dir}"
            )
    else:
        tags = [parsed_args.weight_tag]

    all_results = {}
    base_save_path = (
        os.path.join(weight_dir, "batch_predictions")
        if parsed_args.is_savenii
        else None
    )
    for tag in tags:
        print(f"Testing Stage-1 checkpoint: {tag}", flush=True)
        try:
            random.seed(parsed_args.seed)
            np.random.seed(parsed_args.seed)
            torch.manual_seed(parsed_args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(parsed_args.seed)
            load_label_vae_checkpoint(
                label_encoder,
                label_decoder,
                weight_dir,
                tag,
                device,
                num_classes=parsed_args.num_classes,
                image_size=parsed_args.img_size,
                label_condition_mode=parsed_args.label_condition_mode,
                load_decoder=True,
            )
            test_save_path = None
            if base_save_path is not None:
                test_save_path = os.path.join(base_save_path, tag)
                os.makedirs(test_save_path, exist_ok=True)
            result = inference.evaluate_label_vae(
                dataloader=dataloader,
                label_encoder=label_encoder,
                label_decoder=label_decoder,
                device=device,
                num_classes=parsed_args.num_classes,
                test_save_path=test_save_path,
            )
            all_results[tag] = result
            print(
                f"Stage-1 {tag}: Dice All_10={result.get('mean_dice_all_10', 0):.4f}, "
                "Dice Retina_9_No_OpticDisc="
                f"{result.get('mean_dice_retina_9_no_optic_disc', 0):.4f}",
                flush=True,
            )
        except Exception as exc:
            print(f"Failed to test Stage-1 {tag}: {exc}", flush=True)
            all_results[tag] = None

    excel_output_path = parsed_args.excel_name
    if not os.path.isabs(excel_output_path):
        excel_output_path = os.path.join(weight_dir, excel_output_path)
    save_results_to_excel(all_results, excel_output_path, parsed_args.num_classes)
    print(f"Stage-1 Excel saved to: {excel_output_path}", flush=True)
    if parsed_args.test_all:
        save_selected_stage1_checkpoint(all_results, weight_dir, excel_output_path)
    else:
        print(
            "Single-checkpoint Stage-1 test completed; global Dice selection metadata "
            "was not changed. Use --test_all before Stage-2 training.",
            flush=True,
        )


def evaluate(parsed_args, device):
    if parsed_args.train_stage == "label_vae":
        return evaluate_stage1_label_vae(parsed_args, device)
    _, dataloader = make_dataloaders(parsed_args)
    sample_batch = next(iter(dataloader))
    label_encoder, label_decoder, image_encoder, denoiser = load_components(
        sample_batch,
        parsed_args,
        device,
    )

    weight_dir = parsed_args.pth_path or parsed_args.output_dir
    weight_dir = (
        weight_dir if os.path.isabs(weight_dir) else resolve_project_path(weight_dir)
    )
    contract_path = validate_stage2_encoder_contract_if_present(
        weight_dir,
        label_encoder,
        num_classes=parsed_args.num_classes,
        image_size=parsed_args.img_size,
        label_condition_mode=parsed_args.label_condition_mode,
    )
    if contract_path is not None:
        print(f"Validated test encoder contract: {contract_path}", flush=True)
    if parsed_args.test_all:
        tags = discover_checkpoint_tags(
            weight_dir,
            min_epoch=parsed_args.min_test_epoch,
            max_epoch=parsed_args.max_test_epoch,
            include_best_checkpoint=parsed_args.include_best_checkpoint,
        )
        if not tags:
            raise FileNotFoundError(f"No checkpoint triplets found in {weight_dir}")
    else:
        _, selected_tag, _ = resolve_stage2_resume_checkpoint(
            weight_dir,
            parsed_args.weight_tag,
        )
        tags = [selected_tag]
        if str(parsed_args.weight_tag).strip().lower() == "latest":
            print(f"Resolved latest complete checkpoint: {selected_tag}", flush=True)

    all_results = {}
    base_save_path = (
        os.path.join(weight_dir, "batch_predictions")
        if parsed_args.is_savenii
        else None
    )
    for tag in tags:
        print(f"Testing checkpoint: {tag}", flush=True)
        try:
            random.seed(parsed_args.seed)
            np.random.seed(parsed_args.seed)
            torch.manual_seed(parsed_args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(parsed_args.seed)
            load_checkpoint_tag(
                label_encoder,
                image_encoder,
                denoiser,
                label_decoder,
                weight_dir,
                tag,
                device,
            )
            test_save_path = None
            if base_save_path is not None:
                test_save_path = os.path.join(base_save_path, tag)
                os.makedirs(test_save_path, exist_ok=True)
            result = run_segment(
                parsed_args,
                dataloader,
                label_encoder,
                label_decoder,
                image_encoder,
                denoiser,
                device,
                test_save_path,
            )
            all_results[tag] = result
            print(
                f"{tag}: Dice All_10={result.get('mean_dice_all_10', 0):.4f}, "
                "Dice Retina_9_No_OpticDisc="
                f"{result.get('mean_dice_retina_9_no_optic_disc', 0):.4f}",
                flush=True,
            )
        except Exception as exc:
            print(f"Failed to test {tag}: {exc}", flush=True)
            all_results[tag] = None

    excel_output_path = parsed_args.excel_name
    if not os.path.isabs(excel_output_path):
        excel_output_path = os.path.join(weight_dir, excel_output_path)
    save_results_to_excel(all_results, excel_output_path, parsed_args.num_classes)
    print(f"Excel saved to: {excel_output_path}", flush=True)


def main(parsed_args=None):
    if parsed_args is None:
        parsed_args = build_parser().parse_args()
    parsed_args.data_path = resolve_project_path(parsed_args.data_path)
    parsed_args.list_dir = resolve_project_path(parsed_args.list_dir)
    parsed_args.output_dir = resolve_project_path(parsed_args.output_dir)
    device = setup_runtime(parsed_args)
    if use_training_mode(parsed_args):
        train(parsed_args, device)
    else:
        evaluate(parsed_args, device)
