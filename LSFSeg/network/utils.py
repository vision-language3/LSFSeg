"""General tensor, noise schedule, device, and plotting helpers for LSFSeg."""

import random
import subprocess as sp

import matplotlib.pyplot as plt
import numpy as np
import torch

# -----------------------------
# GPU utilities
# -----------------------------


def set_device(multiple: bool = True):
    """Select GPU device(s) for PyTorch.
    If multiple==False, inspects `nvidia-smi` to pick the GPU with most free memory
    and sets that as the current CUDA device. Otherwise prints available CUDA devices.

    Note: calling this after Torch initialization may not affect CUDA_VISIBLE_DEVICES in some setups.
    """
    if not torch.cuda.is_available():
        print("\nCUDA is not available, running on CPU...\n")
        return torch.device("cpu")

    ngpus = torch.cuda.device_count()
    if multiple:
        print("\nAvailable CUDA devices:")
        for i in range(ngpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return torch.device("cuda")

    # single best GPU selection
    try:
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv"
        out = sp.check_output(cmd.split()).decode("ascii").strip().split("\n")[1:]
        mem_free = [int(x.split()[0]) for x in out]
        best = int(np.argmax(mem_free))
        torch.cuda.set_device(best)
        print(f"\nSelected GPU {best}: {torch.cuda.get_device_name(best)}")
        return torch.device(f"cuda:{best}")
    except Exception:
        # fallback: choose device 0
        print("\nCould not query nvidia-smi, defaulting to cuda:0")
        torch.cuda.set_device(0)
        return torch.device("cuda:0")


# -----------------------------
# Normalization / noise helpers
# -----------------------------


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def standardize(img):
    """Scale image to [0,1]. Accepts numpy array or torch tensor; returns same type as input."""
    is_torch = isinstance(img, torch.Tensor)
    a = img.detach().cpu().numpy() if is_torch else np.array(img)
    a = a - a.min()
    maxv = a.max()
    if maxv <= 0:
        out = a
    else:
        out = a / maxv
    return torch.from_numpy(out) if is_torch else out


def add_noise(img, sigma: float):
    """Add Gaussian noise with standard deviation sigma. Returns same type as input."""
    if isinstance(img, torch.Tensor):
        noise = torch.randn_like(img) * float(sigma)
        out = img + noise
        return standardize(out) if not isinstance(out, np.ndarray) else out
    else:
        noise = np.random.normal(scale=sigma, size=np.shape(img))
        return standardize(img + noise)


def mean_variance_normalization(x, scale: float = 1.0):
    """Zero-mean unit-variance normalization and scaling. Works with numpy or torch."""
    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        mean = torch.mean(x)
        std = torch.std(x)
        out = (x - mean) / (std + 1e-5)
        out = out * float(scale)
        return out
    else:
        mean = np.mean(x)
        std = np.std(x)
        out = (x - mean) / (std + 1e-5)
        out = out * scale
        return out


# -----------------------------
# Noise scheduling helpers
# -----------------------------


def cosine_schedule(t, T, s: float = 0.008):
    """Cosine schedule helper (works on scalars or numpy arrays)."""
    t = np.array(t, dtype=np.float64)
    return np.cos(((t / T + s) / (1 + s)) * (np.pi / 2)) ** 2


def noise_scheduler(T_all: int = 1000, total_timesteps: int = 15):
    """Return (alpha_cumprod, alphas, betas, times) using cosine schedule like original code."""
    times = np.linspace(0, T_all - 1, total_timesteps)
    alpha_cumprod = cosine_schedule(times, T_all) / cosine_schedule(
        np.zeros_like(times), T_all
    )
    alpha_cumprod1 = np.append(1.0, alpha_cumprod[:-1])
    alphas = alpha_cumprod / alpha_cumprod1
    betas = 1 - alphas
    return alpha_cumprod, alphas, betas, times


# Backward-compatible aliases for older code.
cosineFunc = cosine_schedule
noiseScheduler = noise_scheduler


# -----------------------------
# Plotting helpers (work with numpy or torch)
# -----------------------------


def _ensure_2d_image(img):
    """Return a 2D image array for plotting. Accepts (H,W), (H,W,C), torch or numpy."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        return arr.squeeze()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        # channel-first -> HWC
        return np.transpose(arr, (1, 2, 0)).squeeze()
    return arr


def show_mask(mask, alpha=None, ax=None):
    """
    Overlay a colored mask with fixed colors, no transparency.
    mask: 2D (H,W) with integer labels 0~8
    """
    mask = _to_numpy(mask)
    if mask.ndim == 3:
        mask = mask.squeeze()

    color_map = {
        0: [0, 0, 0],
        1: [173, 216, 230],
        2: [0, 255, 255],
        3: [0, 128, 0],
        4: [255, 255, 0],
        5: [255, 165, 0],
        6: [255, 0, 0],
        7: [139, 0, 0],
        8: [0, 255, 0],
    }

    h, w = mask.shape
    mask_image = np.zeros((h, w, 3), dtype=np.float32)

    for k, v in color_map.items():
        color = np.array(v, dtype=np.float32) / 255.0
        mask_image[mask == k] = color

    if ax is not None:
        ax.imshow(mask_image)
    else:
        plt.imshow(mask_image)


def plot_data(img_array, lbl_array):
    idx = random.randint(0, len(img_array) - 1)
    img = _ensure_2d_image(img_array[idx])
    plt.imshow(img)
    show_mask(
        np.where(_to_numpy(lbl_array[idx]) == 1, 1, 0), alpha=0.35, random_color=4
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_sampling(hm_intermediates, idx: int = 0):
    sampling_steps = len(hm_intermediates)
    display_steps = min(10, sampling_steps)
    steps = np.linspace(0, len(hm_intermediates) - 1, display_steps).astype(np.uint64)
    plt.figure(figsize=(3 * display_steps, 5))
    for i, s in enumerate(steps):
        img = _ensure_2d_image(hm_intermediates[int(s)][idx])
        plt.subplot(1, display_steps, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    plt.suptitle("MeanFlow Sampling", fontsize=5 * display_steps)
    plt.tight_layout()
    plt.show()


def plot_noise_parameters(
    times,
    alphas_cumprod,
    betas,
    alphas,
    schedule: str = "cosine",
    sampler: bool = False,
):
    fontsize = 15
    plt.figure(figsize=(10, 3))
    if not sampler:
        plt.subplot(1, 3, 1)
        plt.plot(times, _to_numpy(alphas_cumprod))
        plt.title(r"$\overline{\alpha}$", fontsize=fontsize)
        plt.xlabel("timesteps")
        plt.subplot(1, 3, 2)
        plt.plot(times, _to_numpy(betas))
        plt.title(r"$\beta$", fontsize=fontsize)
        plt.xlabel("timesteps")
        plt.subplot(1, 3, 3)
        plt.plot(times, _to_numpy(alphas))
        plt.title(r"$\alpha$", fontsize=fontsize)
        plt.xlabel("timesteps")
        plt.suptitle(
            "Noise Parameters [schedule: " + schedule + "]", fontsize=fontsize + 5
        )
        plt.tight_layout()
    else:
        plt.subplot(1, 3, 1)
        plt.scatter(times, _to_numpy(alphas_cumprod))
        plt.title(r"$\overline{\alpha}$", fontsize=fontsize)
        plt.xlabel("timesteps")
        plt.xlim(np.max(times), 0)
        plt.subplot(1, 3, 2)
        plt.scatter(times, _to_numpy(betas))
        plt.title(r"$\beta$", fontsize=fontsize)
        plt.xlabel("timesteps")
        plt.xlim(np.max(times), 0)
        plt.subplot(1, 3, 3)
        plt.scatter(times, _to_numpy(alphas))
        plt.title(r"$\alpha$", fontsize=fontsize)
        plt.xlabel("timesteps")
        plt.xlim(np.max(times), 0)
        plt.suptitle(
            "Sampling Parameters [schedule: " + schedule + "]", fontsize=fontsize + 5
        )
    plt.tight_layout()
    plt.show()


def plot_seg(img, prediction, nplot: int = 3):
    idx = random.sample(range(0, len(img)), nplot)
    plt.figure(figsize=(5 * nplot, 5))
    for i, ii in enumerate(idx):
        plt.subplot(1, nplot, i + 1)
        plt.imshow(_ensure_2d_image(img[ii]))
        show_mask(np.where(_to_numpy(prediction[ii]) == 1, 1, 0))
        plt.title(f"Img-{ii:d}", fontsize=12)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
