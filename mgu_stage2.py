"""Train or test the LSFSeg Stage-2 MeanFlow segmentation model."""

from __future__ import annotations

import argparse
import os

from network.config import (
    FOLD_NAME,
    IMAGE_SIZE,
    NUM_CLASSES,
    fold_output_dir,
    validate_mgu_layout,
)
from network.pipeline import build_parser, main, resolve_selected_stage1_checkpoint

DEFAULT_DATA_PATH = os.path.join("Data", "MGU")
DEFAULT_LIST_DIR = os.path.join(DEFAULT_DATA_PATH, "lists")
DEFAULT_STAGE1_OUTPUT_DIR = os.path.join("output", "MGU", "single_split", "stage1")
DEFAULT_OUTPUT_DIR = os.path.join("output", "MGU", "single_split", "stage2")
DEFAULT_EXCEL_NAME = f"{FOLD_NAME}_MGU_results.xlsx"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train or test the LSFSeg Stage-2 segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("train", "test"), default="test")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--list_dir", default=DEFAULT_LIST_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--stage1_output_dir", default=DEFAULT_STAGE1_OUTPUT_DIR)
    parser.add_argument("--additional_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--lr_decay_epochs", type=int, default=20)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5)
    parser.add_argument("--segmentation_weight", type=float, default=5.0)
    parser.add_argument("--dual_view_consistency_weight", type=float, default=0.05)
    parser.add_argument("--dual_view_latent_weight", type=float, default=0.25)
    parser.add_argument("--dual_view_warmup_epochs", type=int, default=10)
    parser.add_argument("--resume_dir", default=None)
    parser.add_argument("--resume_tag", default="latest")

    parser.add_argument("--weight_tag", default="latest")
    parser.add_argument(
        "--test_all",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--min_test_epoch", type=int, default=0)
    parser.add_argument("--max_test_epoch", type=int, default=None)
    parser.add_argument("--excel_name", default=DEFAULT_EXCEL_NAME)
    parser.add_argument(
        "--save_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_denoise_visuals",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def resolve_list_dir(args):
    return (
        args.list_dir
        or os.environ.get("LIST_DIR")
        or os.path.join(args.data_path, "lists")
    )


def build_train_args(args, list_dir):
    stage1_dir = args.stage1_output_dir or fold_output_dir("stage1")
    output_dir = args.output_dir or fold_output_dir("stage2")
    stage1_weight_tag = resolve_selected_stage1_checkpoint(stage1_dir)
    parser = build_parser()
    parser.set_defaults(
        dataset="MGU",
        train=True,
        eval=False,
        test_all=False,
        train_stage="flow",
        data_path=args.data_path,
        list_dir=list_dir,
        output_dir=output_dir,
        label_vae_dir=stage1_dir,
        weight_tag=stage1_weight_tag,
        img_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        batch_size=args.batch_size,
        seed=123,
        device=args.device,
        max_epochs=args.additional_epochs,
        stage2_additional_epochs=args.additional_epochs,
        base_lr=args.lr,
        stage2_lr_decay_epochs=args.lr_decay_epochs,
        stage2_lr_decay_gamma=args.lr_decay_gamma,
        label_condition_mode="one_hot",
        latent_fusion_hidden_channels=128,
        p_mean=0.8,
        p_std=0.8,
        noise_scale=1.0,
        t_eps=0.05,
        ema_decay=0.999,
        segmentation_weight=args.segmentation_weight,
        denoise_weight=1.0,
        meanflow_data_proportion=0.5,
        meanflow_norm_p=1.0,
        meanflow_norm_eps=0.01,
        dual_view_consistency_weight=args.dual_view_consistency_weight,
        dual_view_latent_weight=args.dual_view_latent_weight,
        dual_view_warmup_epochs=args.dual_view_warmup_epochs,
        stage2_resume_dir=args.resume_dir,
        stage2_resume_tag=args.resume_tag,
        sampler="meanflow",
        sampling_steps=1,
        n_eval=1,
        sigma=0.0,
        excel_name=f"{FOLD_NAME}_MGU_results.xlsx",
    )
    return parser.parse_args([])


def build_test_args(args, list_dir):
    weight_dir = args.output_dir or fold_output_dir("stage2")
    excel_name = args.excel_name or f"{FOLD_NAME}_MGU_results.xlsx"
    parser = build_parser()
    parser.set_defaults(
        dataset="MGU",
        train=False,
        eval=True,
        test_all=args.test_all,
        train_stage="flow",
        weight_tag=args.weight_tag,
        data_path=args.data_path,
        list_dir=list_dir,
        output_dir=weight_dir,
        pth_path=weight_dir,
        excel_name=excel_name,
        img_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        batch_size=args.batch_size,
        seed=123,
        device=args.device,
        is_savenii=args.save_predictions,
        save_denoise_visuals=args.save_denoise_visuals,
        label_condition_mode="one_hot",
        latent_fusion_hidden_channels=128,
        latent_visual_t=0.95,
        sampler="meanflow",
        sampling_steps=1,
        n_eval=1,
        sigma=0.0,
        noise_scale=1.0,
        t_eps=0.05,
        min_test_epoch=args.min_test_epoch,
        max_test_epoch=args.max_test_epoch,
        include_best_checkpoint=True,
    )
    return parser.parse_args([])


def validate_args(args):
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if args.mode == "train":
        if args.additional_epochs < 1:
            raise ValueError("--additional_epochs must be at least 1")
        if args.lr <= 0.0:
            raise ValueError("--lr must be positive")
        if args.lr_decay_epochs < 1:
            raise ValueError("--lr_decay_epochs must be at least 1")
        if not 0.0 < args.lr_decay_gamma <= 1.0:
            raise ValueError("--lr_decay_gamma must be in (0, 1]")
        if args.segmentation_weight < 0.0:
            raise ValueError("--segmentation_weight must be non-negative")
        if args.dual_view_consistency_weight < 0.0:
            raise ValueError("--dual_view_consistency_weight must be non-negative")
        if args.dual_view_latent_weight < 0.0:
            raise ValueError("--dual_view_latent_weight must be non-negative")
        if args.dual_view_warmup_epochs < 0:
            raise ValueError("--dual_view_warmup_epochs must be non-negative")
        output_dir = args.output_dir or fold_output_dir("stage2")
        if args.resume_dir and os.path.normcase(
            os.path.abspath(args.resume_dir)
        ) == os.path.normcase(os.path.abspath(output_dir)):
            raise ValueError(
                "--output_dir must differ from --resume_dir so loaded weights "
                "remain read-only"
            )
    elif args.save_denoise_visuals and not args.save_predictions:
        raise ValueError(
            "--save_denoise_visuals requires --save_predictions because both use "
            "the per-checkpoint prediction directory"
        )


def run(argv=None):
    args = parse_args(argv)
    validate_args(args)
    list_dir = resolve_list_dir(args)
    validate_mgu_layout(args.data_path, list_dir)
    if args.mode == "train":
        main(build_train_args(args, list_dir))
        return
    main(build_test_args(args, list_dir))


if __name__ == "__main__":
    run()
