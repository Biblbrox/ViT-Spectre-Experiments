# %%

import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# %%


def load_nuscenes_images(nuscenes_root, camera="CAM_FRONT"):
    """
    Collect all image paths from a specific nuScenes camera folder.

    Example expected structure:
    nuscenes_root/
        samples/
            CAM_FRONT/
                *.jpg
    """
    img_dir = os.path.join(nuscenes_root, "samples", camera)
    image_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {img_dir}")
    return image_paths


def compute_fft(image_gray):
    """
    Compute 2D FFT and return shifted FFT and magnitude spectrum.
    """
    fft = np.fft.fft2(image_gray).real
    return fft


def extract_frequencies(magnitude):
    """
    Compute frequency grid and corresponding magnitudes.
    """
    h, w = magnitude.shape

    fx = np.fft.fftfreq(w)
    fy = np.fft.fftfreq(h)
    fx_grid, fy_grid = np.meshgrid(fx, fy)

    freq_radius = np.sqrt(fx_grid**2 + fy_grid**2)

    return freq_radius.flatten(), magnitude.flatten()


def plot_frequency_spectrum(freqs, mags, title):
    """
    Scatter plot of frequency vs magnitude.
    """
    plt.figure(figsize=(8, 6))
    # plt.scatter(freqs, np.log1p(mags), s=1)
    plt.scatter(freqs, mags, s=1)
    plt.yscale("log")
    plt.xlabel("Frequency radius")
    plt.ylabel("Log Magnitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_image_fft(nuscenes_root, image_index, camera="CAM_FRONT"):
    """
    Full pipeline:
    - Load image by index
    - Compute FFT
    - Extract frequencies
    - Plot spectrum
    """
    image_paths = load_nuscenes_images(nuscenes_root, camera)

    if image_index >= len(image_paths):
        raise IndexError("Image index out of range")

    img_path = image_paths[image_index]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Failed to load image: {img_path}")

    fft = compute_fft(img)
    freqs, mags = extract_frequencies(fft)

    plot_frequency_spectrum(freqs, mags, title=f"FFT Frequency Spectrum (Index {image_index})")

    return freqs, mags


# %%

nuscenes_root = "/storage/experiments-ml/datasets/nuscenes"
image_index = 120
analyze_image_fft(nuscenes_root, image_index)


# ### Visualize MNIST FFT spectre

# %%

fft = torch.load("fft_tensor.pt").detach().cpu()

# Visualize the FFT tensor
print(fft.shape)  # [B, N, D]

# plt.imshow(fft[0, 0, :].reshape(1, 16), cmap='gray')
for i in range(fft.shape[0]):
    plt.hist(fft[i, 0, :].reshape(1, 16))
