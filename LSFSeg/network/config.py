"""Shared MGU dataset layout and label convention."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = (
    "background",
    "RNFL",
    "GCL",
    "IPL",
    "INL",
    "OPL",
    "ONL",
    "IS/OS",
    "RPE",
    "Choroid",
    "Optic_disc",
)

CLASS_COLORS = (
    (0, 0, 0),
    (0, 136, 255),
    (34, 34, 204),
    (0, 204, 34),
    (255, 221, 0),
    (238, 34, 34),
    (255, 102, 34),
    (0, 221, 221),
    (170, 68, 204),
    (119, 34, 153),
    (232, 68, 112),
)


def project_path(*parts: str) -> str:
    """Build a path relative to the LSFSeg project directory."""
    return os.path.join(*parts)


DATA_PATH = os.environ.get("DATA_PATH", project_path("Data", "MGU"))
LIST_DIR = os.environ.get("LIST_DIR", os.path.join(DATA_PATH, "lists"))
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", project_path("output", "MGU"))
FOLD_NAME = os.environ.get("FOLD_NAME", "single_split")

IMAGE_SIZE = [512, 512]
ORIGINAL_IMAGE_SIZE = [992, 1024]
NUM_CLASSES = 11
SELECTED_STAGE1_JSON = "selected_stage1.json"
SELECTED_STAGE1_TXT = "selected_stage1_tag.txt"
EXPECTED_SPLIT_SIZES = {"train": 148, "val": 48, "test": 48}


def fold_output_dir(*parts: str) -> str:
    return os.path.join(OUTPUT_ROOT, FOLD_NAME, *parts)


def validate_mgu_layout(
    data_path: str = DATA_PATH,
    list_dir: str = LIST_DIR,
) -> None:
    """Validate the source-resolution MGU images, labels, and split files."""

    root = Path(data_path)
    lists = Path(list_dir)
    image_dir = root / "image"
    label_dir = root / "label"
    required_directories = [image_dir, label_dir]
    missing = [str(path) for path in required_directories if not path.is_dir()]
    split_paths = [lists / f"{split}.txt" for split in ("train", "val", "test")]
    missing.extend(str(path) for path in split_paths if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "MGU layout is incomplete. Expected image/, label/, and "
            f"lists/{{train,val,test}}.txt. Missing: {missing}"
        )

    split_entries: dict[str, list[str]] = {}
    names: list[str] = []
    for split, split_path in zip(("train", "val", "test"), split_paths):
        split_names = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not split_names:
            raise ValueError(f"MGU split is empty: {split_path}")
        if len(split_names) != len(set(split_names)):
            raise ValueError(f"MGU split contains duplicate entries: {split_path}")
        expected_count = EXPECTED_SPLIT_SIZES[split]
        if len(split_names) != expected_count:
            raise ValueError(
                f"MGU {split} split must contain {expected_count} samples, got {len(split_names)}"
            )
        split_entries[split] = split_names
        names.extend(split_names)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(set(split_entries[left]) & set(split_entries[right]))
        if overlap:
            raise ValueError(
                f"MGU {left}/{right} splits overlap; first entries: {overlap[:10]}"
            )

    unique_names = list(dict.fromkeys(names))
    sample_directories = [image_dir, label_dir]
    missing_samples = [
        name
        for name in unique_names
        if not all((directory / name).is_file() for directory in sample_directories)
    ]
    if missing_samples:
        preview = missing_samples[:10]
        raise FileNotFoundError(
            f"{len(missing_samples)} split entries have no matching source-resolution "
            f"image/label pair; first entries: {preview}."
        )

    image_files = {path.name for path in image_dir.iterdir() if path.is_file()}
    label_files = {path.name for path in label_dir.iterdir() if path.is_file()}
    listed_files = set(unique_names)
    unexpected_images = sorted(image_files - listed_files)
    unexpected_labels = sorted(label_files - listed_files)
    if unexpected_images or unexpected_labels:
        raise ValueError(
            "MGU data directories contain files outside the single split: "
            f"images={unexpected_images[:10]}, labels={unexpected_labels[:10]}"
        )

    observed: set[int] = set()
    for name in unique_names:
        if Path(name).suffix.lower() != ".png":
            raise ValueError(f"MGU image/label files must be PNG: {name}")
        with Image.open(image_dir / name) as image_handle:
            image_size = image_handle.size
            image_mode = image_handle.mode
        with Image.open(label_dir / name) as label_handle:
            label_size = label_handle.size
            label_mode = label_handle.mode
            label_palette = label_handle.getpalette()
            label = np.asarray(label_handle)
        expected_original_size = (ORIGINAL_IMAGE_SIZE[1], ORIGINAL_IMAGE_SIZE[0])
        if image_size != expected_original_size or label_size != expected_original_size:
            raise ValueError(
                f"MGU image/label must preserve original size {expected_original_size} for {name}: "
                f"image={image_size}, label={label_size}"
            )
        if image_mode != "L" or label_mode != "P":
            raise ValueError(
                "MGU original PNG modes must be image=L and label=P for "
                f"{name}: image={image_mode}, label={label_mode}"
            )
        expected_palette = [channel for color in CLASS_COLORS for channel in color]
        if (
            label_palette is None
            or label_palette[: len(expected_palette)] != expected_palette
        ):
            raise ValueError(
                f"MGU label palette does not match the configured class colors: {label_dir / name}"
            )
        if label.ndim != 2:
            raise ValueError(
                f"MGU label must be a single-channel categorical mask: {label_dir / name}"
            )
        observed.update(int(value) for value in np.unique(label))
    invalid = sorted(value for value in observed if value < 0 or value >= NUM_CLASSES)
    if invalid:
        raise ValueError(
            f"MGU labels contain ids outside 0..{NUM_CLASSES - 1}: {invalid}. "
            f"Expected {dict(enumerate(CLASS_NAMES))}."
        )
    expected_ids = set(range(NUM_CLASSES))
    if observed != expected_ids:
        raise ValueError(
            f"MGU single split must contain label ids 0..{NUM_CLASSES - 1}, got {sorted(observed)}"
        )
    counts = {split: len(entries) for split, entries in split_entries.items()}
    print(
        f"MGU preflight passed: split_sizes={counts}, total={len(unique_names)}, "
        f"observed labels={sorted(observed)}, validated_inputs=source-resolution",
        flush=True,
    )
