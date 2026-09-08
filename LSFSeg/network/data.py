"""Dataset and augmentation utilities for LSFSeg.

This module prepares MGU OCT images, categorical labels, and the label
condition tensor consumed by the label encoder.
"""

import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter, rotate
from torch.utils.data import Dataset


def random_image_intensity(image, noise_std=0.12):
    """Apply image-only intensity augmentation on [0, 1] MGU images."""
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    if random.random() < 0.5:
        noise = np.random.normal(0.0, noise_std * 255.0, image_uint8.shape).astype(
            np.float32
        )
        image_uint8 = np.clip(image_uint8.astype(np.float32) + noise, 0, 255).astype(
            np.uint8
        )

    image_pil = Image.fromarray(image_uint8)
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        image_pil = ImageEnhance.Brightness(image_pil).enhance(brightness_factor)
        image_pil = ImageEnhance.Contrast(image_pil).enhance(contrast_factor)

    return np.asarray(image_pil, dtype=np.float32) / 255.0


def random_strong_image_intensity(image):
    """Photometric-only strong OCT augmentation for dual-view consistency."""
    image = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)

    gamma = random.uniform(0.65, 1.55)
    image = np.power(np.clip(image, 1e-6, 1.0), gamma)

    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image_pil = Image.fromarray(image_uint8)
    image_pil = ImageEnhance.Brightness(image_pil).enhance(random.uniform(0.65, 1.35))
    image_pil = ImageEnhance.Contrast(image_pil).enhance(random.uniform(0.60, 1.45))
    image = np.asarray(image_pil, dtype=np.float32) / 255.0

    speckle_std = random.uniform(0.08, 0.22)
    speckle = np.random.normal(1.0, speckle_std, image.shape).astype(np.float32)
    image = image * speckle

    height, width = image.shape
    depth = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1)
    attenuation = np.exp(-random.uniform(0.05, 0.35) * depth)
    image = image * attenuation

    center = random.uniform(0.15 * width, 0.85 * width)
    shadow_width = random.uniform(0.06 * width, 0.18 * width)
    shadow_strength = random.uniform(0.08, 0.30)
    columns = np.arange(width, dtype=np.float32)
    shadow = np.exp(-0.5 * ((columns - center) / max(1.0, shadow_width)) ** 2)
    image = image * (1.0 - shadow_strength * shadow.reshape(1, width))

    if random.random() < 0.5:
        image = gaussian_filter(image, sigma=random.uniform(0.35, 0.80))

    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def normalize_image(image):
    """Normalize one OCT B-scan before feeding it into the network."""
    image = image.astype(np.float32, copy=False)
    mean = float(np.mean(image))
    std = float(np.std(image))
    return (image - mean) / (std + 1e-5)


def resize_image_and_label(image, label, output_size):
    """Resize an OCT image and categorical mask without interpolating class ids."""
    output_size = tuple(int(value) for value in output_size)
    if len(output_size) != 2 or min(output_size) < 1:
        raise ValueError(
            f"output_size must contain two positive values, got {output_size}"
        )
    if image.ndim != 2 or label.ndim != 2 or image.shape != label.shape:
        raise ValueError(
            "MGU image and label must be matching 2D arrays before resize: "
            f"image={image.shape}, label={label.shape}"
        )
    if image.shape == output_size:
        return (
            image.astype(np.float32, copy=False),
            label.astype(np.int64, copy=False),
        )
    if label.size and (int(label.min()) < 0 or int(label.max()) > 255):
        raise ValueError(
            "MGU label ids must fit in uint8 before nearest-neighbor resize: "
            f"min={int(label.min())}, max={int(label.max())}"
        )

    target_width_height = (output_size[1], output_size[0])
    image_uint8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    label_uint8 = label.astype(np.uint8, copy=False)
    resized_image = (
        np.array(
            Image.fromarray(image_uint8, mode="L").resize(
                target_width_height,
                resample=Image.Resampling.BICUBIC,
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    resized_label = np.array(
        Image.fromarray(label_uint8, mode="L").resize(
            target_width_height,
            resample=Image.Resampling.NEAREST,
        ),
        dtype=np.int64,
    )
    return resized_image, resized_label


def random_horizontal_flip(image, label):
    image = np.flip(image, axis=1).copy()
    label = np.flip(label, axis=1).copy()
    return image, label


def horizontal_flip_image(image):
    return np.flip(image, axis=1).copy()


def angle_rotate_image(image, angle):
    return rotate(image, angle, reshape=False, order=1, mode="constant", cval=0)


def angle_rotate_label(label, angle):
    return rotate(label, angle, reshape=False, order=0, mode="constant", cval=0)


def random_angle_rotate(image, label, max_angle=20):
    """Apply random angle rotation to an MGU image and categorical label."""
    angle = np.random.uniform(-max_angle, max_angle)
    image = angle_rotate_image(image, angle)
    label = angle_rotate_label(label, angle)
    return image, label


def normalize_label_condition_mode(mode="one_hot"):
    """Validate the final one-hot label-encoder input format."""
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"one_hot", "onehot"}:
        return "one_hot"
    raise ValueError(
        "Unsupported label conditioning mode. Expected one_hot. "
        f"Got label_condition_mode='{mode}'."
    )


def build_label_condition(label, num_classes=None, mode="one_hot"):
    """Build the final one-hot label-encoder input tensor."""
    normalize_label_condition_mode(mode)
    label_map = label.astype(np.int64, copy=False)
    if label_map.ndim != 2:
        raise ValueError(
            "Label condition must be built from a 2D categorical map, "
            f"got shape {tuple(label_map.shape)}."
        )
    if num_classes is not None and label_map.size > 0:
        min_label = int(label_map.min())
        max_label = int(label_map.max())
        if min_label < 0 or max_label >= int(num_classes):
            raise ValueError(
                "Label contains class ids outside the configured range "
                f"[0, {int(num_classes) - 1}]: min={min_label}, max={max_label}."
            )
    class_count = (
        int(num_classes) if num_classes is not None else int(label_map.max()) + 1
    )
    one_hot = np.eye(class_count, dtype=np.float32)[label_map]
    one_hot = np.moveaxis(one_hot, -1, 0)
    return one_hot


class RandomGenerator:
    """Online resize and augmentation for Stage1 and Stage2 training."""

    def __init__(
        self,
        output_size,
        num_classes=11,
        label_condition_mode="one_hot",
        enable_dual_view=False,
    ):
        self.output_size = tuple(output_size)
        self.num_classes = int(num_classes)
        self.label_condition_mode = normalize_label_condition_mode(label_condition_mode)
        self.enable_dual_view = bool(enable_dual_view)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image, label = resize_image_and_label(image, label, self.output_size)

        image_weak = random_image_intensity(image)
        image_strong = (
            random_strong_image_intensity(image) if self.enable_dual_view else None
        )

        if random.random() > 0.5:
            angle = np.random.uniform(-20, 20)
            image_weak = angle_rotate_image(image_weak, angle)
            if image_strong is not None:
                image_strong = angle_rotate_image(image_strong, angle)
            label = angle_rotate_label(label, angle)

        if random.random() > 0.5:
            image_weak = horizontal_flip_image(image_weak)
            if image_strong is not None:
                image_strong = horizontal_flip_image(image_strong)
            label = horizontal_flip_image(label)

        if image_weak.shape != self.output_size or label.shape != self.output_size:
            raise RuntimeError(
                "Online MGU resize produced an unexpected shape: "
                f"image={image_weak.shape}, label={label.shape}, expected={self.output_size}"
            )
        if image_strong is not None and image_strong.shape != self.output_size:
            raise ValueError(
                "MGU strong augmentation must preserve the 512x512 model input size: "
                f"image_strong={image_strong.shape}, expected={self.output_size}"
            )

        image_weak = normalize_image(image_weak)
        if image_strong is not None:
            image_strong = normalize_image(image_strong)
        label = label.astype(np.int64, copy=False)
        condition = build_label_condition(
            label, self.num_classes, self.label_condition_mode
        )

        output = {
            "image": torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0),
            "label": torch.from_numpy(label),
            "sdf": torch.from_numpy(condition.astype(np.float32, copy=False)),
        }
        if image_strong is not None:
            output["image_strong"] = torch.from_numpy(
                image_strong.astype(np.float32)
            ).unsqueeze(0)
        return output


class MGUDataset(Dataset):
    """MGU OCT dataset with categorical-map-derived latent conditioning.

    The key name ``sdf`` is kept for compatibility with the existing trainer,
    but it contains the configured label condition, not distance maps.
    Every split reads source-resolution files directly from ``image/label``.
    """

    def __init__(
        self,
        base_dir,
        list_dir,
        split,
        transform=None,
        target_size=(512, 512),
        num_classes=11,
        label_condition_mode="one_hot",
    ):
        self.transform = transform
        self.split = split
        self.target_size = tuple(target_size)
        self.num_classes = int(num_classes)
        self.label_condition_mode = normalize_label_condition_mode(label_condition_mode)

        with open(os.path.join(list_dir, split + ".txt"), encoding="utf-8") as handle:
            self.sample_list = handle.readlines()

        self.image_dir = os.path.join(base_dir, "image")
        self.label_dir = os.path.join(base_dir, "label")
        missing_directories = [
            path for path in (self.image_dir, self.label_dir) if not os.path.isdir(path)
        ]
        if missing_directories:
            raise FileNotFoundError(
                "MGU input directories are missing: " f"{missing_directories}"
            )

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip()
        image_path = os.path.join(self.image_dir, slice_name)
        label_path = os.path.join(self.label_dir, slice_name)

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
        label = np.array(Image.open(label_path), dtype=np.int64)
        source_size = tuple(label.shape)
        if image.shape != label.shape:
            raise ValueError(
                "MGU image and label shapes do not match: "
                f"image={image.shape}, label={label.shape}, case={slice_name}"
            )
        original_label = label.copy() if self.split == "test" else None
        original_size = source_size
        sample = {"image": image, "label": label}

        if self.transform is not None:
            sample = self.transform(sample)
        else:
            image, label = resize_image_and_label(image, label, self.target_size)
            model_image = normalize_image(image)
            condition = build_label_condition(
                label, self.num_classes, self.label_condition_mode
            )
            if self.split == "test":
                output_label = original_label.astype(np.int64, copy=False)
            else:
                output_label = label
            sample = {
                "image": torch.from_numpy(model_image.astype(np.float32)).unsqueeze(0),
                "label": torch.from_numpy(output_label),
                "sdf": torch.from_numpy(condition.astype(np.float32, copy=False)),
            }

        sample["case_name"] = slice_name
        sample["original_size"] = torch.tensor(original_size, dtype=torch.long)
        return sample
