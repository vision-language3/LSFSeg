"""Train or test the LSFSeg Stage-1 label autoencoder."""

from __future__ import annotations

import argparse
import os

from network.config import (
    FOLD_NAME,
    IMAGE_SIZE,
    NUM_CLASSES,
    SELECTED_STAGE1_JSON,
    SELECTED_STAGE1_TXT,
    fold_output_dir,
    validate_mgu_layout,
)
from network.pipeline import build_parser, main

DEFAULT_DATA_PATH = os.path.join("Data", "MGU")
DEFAULT_LIST_DIR = os.path.join(DEFAULT_DATA_PATH, "lists")
DEFAULT_OUTPUT_DIR = os.path.join("output", "MGU", "single_split", "stage1")
DEFAULT_EXCEL_NAME = f"{FOLD_NAME}_MGU_stage1_results.xlsx"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train or test the LSFSeg Stage-1 label autoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("train", "test"), default="test")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--list_dir", default=DEFAULT_LIST_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument(
        "--lr_scheduler",
        choices=("constant", "warmup_linear"),
        default="warmup_linear",
    )
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_interval_steps", type=int, default=5000)

    parser.add_argument("--weight_tag", default="best_epoch")
    parser.add_argument(
        "--test_all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="test every complete Stage-1 checkpoint and update selection metadata",
    )
    parser.add_argument("--min_test_epoch", type=int, default=0)
    parser.add_argument("--max_test_epoch", type=int, default=None)
    parser.add_argument("--excel_name", default=DEFAULT_EXCEL_NAME)
    parser.add_argument(
        "--save_predictions",
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
    output_dir = args.output_dir or fold_output_dir("stage1")
    parser = build_parser()
    parser.set_defaults(
        dataset="MGU",
        train=True,
        eval=False,
        test_all=False,
        train_stage="label_vae",
        data_path=args.data_path,
        list_dir=list_dir,
        output_dir=output_dir,
        img_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        batch_size=args.batch_size,
        seed=123,
        device=args.device,
        max_epochs=args.epochs,
        base_lr=args.lr,
        label_condition_mode="one_hot",
        sampling_steps=1,
        sampler="meanflow",
        n_eval=1,
        sigma=0.0,
        stage1_ce_weight=args.ce_weight,
        stage1_dice_weight=args.dice_weight,
        stage1_kl_weight=args.kl_weight,
        stage1_lr_scheduler=args.lr_scheduler,
        stage1_warmup_steps=args.warmup_steps,
        stage1_max_steps=args.max_steps,
        stage1_save_every=0,
        stage1_save_interval_steps=args.save_interval_steps,
        stage1_grad_clip=args.grad_clip,
    )
    return parser.parse_args([])


def build_test_args(args, list_dir):
    weight_dir = args.output_dir or fold_output_dir("stage1")
    excel_name = args.excel_name or f"{FOLD_NAME}_MGU_stage1_results.xlsx"
    parser = build_parser()
    parser.set_defaults(
        dataset="MGU",
        train=False,
        eval=True,
        test_all=args.test_all,
        train_stage="label_vae",
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
        save_denoise_visuals=False,
        label_condition_mode="one_hot",
        min_test_epoch=args.min_test_epoch,
        max_test_epoch=args.max_test_epoch,
        include_best_checkpoint=True,
    )
    return parser.parse_args([])


def validate_args(args):
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if args.mode == "train":
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1")
        if args.lr <= 0.0:
            raise ValueError("--lr must be positive")
        if args.max_steps is not None and args.max_steps < 1:
            raise ValueError("--max_steps must be at least 1")
        if min(args.ce_weight, args.dice_weight, args.kl_weight) < 0.0:
            raise ValueError("Stage-1 loss weights must be non-negative")
        if args.warmup_steps < 0:
            raise ValueError("--warmup_steps must be non-negative")
        if args.grad_clip < 0.0:
            raise ValueError("--grad_clip must be non-negative")
        if args.save_interval_steps < 0:
            raise ValueError("--save_interval_steps must be non-negative")


def run(argv=None):
    args = parse_args(argv)
    validate_args(args)
    list_dir = resolve_list_dir(args)
    validate_mgu_layout(args.data_path, list_dir)

    if args.mode == "train":
        output_dir = args.output_dir or fold_output_dir("stage1")
        for metadata_name in (SELECTED_STAGE1_JSON, SELECTED_STAGE1_TXT):
            metadata_path = os.path.join(output_dir, metadata_name)
            if os.path.isfile(metadata_path):
                os.remove(metadata_path)
                print(
                    f"Removed stale Stage-1 selection metadata: {metadata_path}",
                    flush=True,
                )
        main(build_train_args(args, list_dir))
        return

    main(build_test_args(args, list_dir))


if __name__ == "__main__":
    run()
