"""

    generate_presentation_results.py - Generates tehe compirison between FFDNet and DnCNN.
    It compares PSNR gains and inference times.

"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import matplotlib.pyplot as plt
from PIL import Image
from includes import *

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from FFDnet.FFDnet import FFDNet, FFDNetConfig
from DnCNN.dnCNN import DnCNN
import dataset


def run_ffdnet(model, noisy_img, sigma, device):
    model.eval()
    noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    
    # Padding for FFDNet (must be divisible by 2)
    _, _, H, W = noisy_tensor.shape
    h_pad = (2 - H % 2) % 2
    w_pad = (2 - W % 2) % 2
    if h_pad > 0 or w_pad > 0:
        noisy_tensor = F_torch.pad(noisy_tensor, (0, w_pad, 0, h_pad), mode='reflect')
    
    sigma_map = torch.full_like(noisy_tensor, sigma/255.0)
    
    start_time = time.time()
    with torch.no_grad():
        predicted_noise = model(noisy_tensor, sigma_map)
        denoised = noisy_tensor - predicted_noise
    end_time = time.time()
    
    # Crop back
    if h_pad > 0 or w_pad > 0:
        denoised = denoised[:, :, :H, :W]
        predicted_noise = predicted_noise[:, :, :H, :W]
        
    return denoised.squeeze().cpu().numpy(), predicted_noise.squeeze().cpu().numpy(), end_time - start_time

def run_dncnn(model, noisy_img, device):
    model.eval()
    noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    
    start_time = time.time()
    with torch.no_grad():
        predicted_noise = model(noisy_tensor)
        denoised = noisy_tensor - predicted_noise
    end_time = time.time()
    
    return denoised.squeeze().cpu().numpy(), predicted_noise.squeeze().cpu().numpy(), end_time - start_time

# --- Visualization ---

def plot_single_model(name, clean, noisy, denoised, predicted_noise, psnr_noisy, psnr_denoised, time_taken, sigma):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    plt.suptitle(f"{name} Analysis (Sigma={sigma}) - Time: {time_taken:.4f}s", fontsize=16, fontweight='bold')
    
    # 1. Noisy Input
    axes[0].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Noisy Input\nPSNR: {psnr_noisy:.2f} dB", fontsize=12)
    axes[0].axis('off')
    
    # 2. Predicted Noise (What the model removed)
    axes[1].imshow(predicted_noise, cmap='gray') # Auto scale for visibility
    axes[1].set_title("Predicted Noise\n(What model removed)", fontsize=12)
    axes[1].axis('off')
    
    # 3. Denoised Output
    axes[2].imshow(denoised, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Denoised Output\nPSNR: {psnr_denoised:.2f} dB (+{psnr_denoised-psnr_noisy:.2f})", fontsize=12, fontweight='bold', color='green')
    axes[2].axis('off')
    
    # 4. Ground Truth
    axes[3].imshow(clean, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title("Ground Truth\n(Original)", fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    filename = f"presentation_{name.lower()}_analysis.png"
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close()

def plot_comparison(clean, noisy, ffdnet_res, dncnn_res, sigma):
    # ffdnet_res/dncnn_res = (denoised, time, psnr)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)
    plt.suptitle(f"Model Comparison (Sigma={sigma})", fontsize=18, fontweight='bold')
    
    # Images Row
    ax_clean = fig.add_subplot(gs[0, 0])
    ax_clean.imshow(clean, cmap='gray', vmin=0, vmax=1)
    ax_clean.set_title("Ground Truth")
    ax_clean.axis('off')
    
    ax_ffdnet = fig.add_subplot(gs[0, 1])
    ax_ffdnet.imshow(ffdnet_res['img'], cmap='gray', vmin=0, vmax=1)
    ax_ffdnet.set_title(f"FFDNet\nPSNR: {ffdnet_res['psnr']:.2f} dB")
    ax_ffdnet.axis('off')
    
    ax_dncnn = fig.add_subplot(gs[0, 2])
    ax_dncnn.imshow(dncnn_res['img'], cmap='gray', vmin=0, vmax=1)
    ax_dncnn.set_title(f"DnCNN\nPSNR: {dncnn_res['psnr']:.2f} dB")
    ax_dncnn.axis('off')
    
    # Metrics Row
    # PSNR Comparison
    ax_psnr = fig.add_subplot(gs[1, 0])
    models = ['Noisy', 'FFDNet', 'DnCNN']
    psnrs = [dataset.calculate_psnr(clean, noisy), ffdnet_res['psnr'], dncnn_res['psnr']]
    colors = ['gray', 'blue', 'orange']
    bars = ax_psnr.bar(models, psnrs, color=colors)
    ax_psnr.set_title("PSNR Quality (Higher is Better)")
    ax_psnr.set_ylabel("dB")
    ax_psnr.bar_label(bars, fmt='%.2f')
    
    # Time Comparison
    ax_time = fig.add_subplot(gs[1, 1])
    times = [ffdnet_res['time'], dncnn_res['time']]
    t_models = ['FFDNet', 'DnCNN']
    t_colors = ['blue', 'orange']
    bars_t = ax_time.bar(t_models, times, color=t_colors)
    ax_time.set_title("Inference Time (Lower is Better)")
    ax_time.set_ylabel("Seconds")
    ax_time.bar_label(bars_t, fmt='%.4f')
    
    # Zoom/Crop Comparison (Center crop)
    H, W = clean.shape
    cy, cx = H//2, W//2
    s = 50 # crop size
    
    crop_clean = clean[cy-s:cy+s, cx-s:cx+s]
    crop_ffdnet = ffdnet_res['img'][cy-s:cy+s, cx-s:cx+s]
    crop_dncnn = dncnn_res['img'][cy-s:cy+s, cx-s:cx+s]
    
    ax_zoom = fig.add_subplot(gs[1, 2])
    # Stitch crops together
    stitched = np.hstack((crop_clean, crop_ffdnet, crop_dncnn))
    ax_zoom.imshow(stitched, cmap='gray', vmin=0, vmax=1)
    ax_zoom.set_title("Detail Zoom: GT | FFDNet | DnCNN")
    ax_zoom.axis('off')
    
    plt.tight_layout()
    filename = "presentation_model_comparison.png"
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close()

# --- Main ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Models
    print("Loading models...")
    
    # FFDNet
    ffdnet = FFDNet(in_channels=1, num_features=64, num_conv_layers=15).to(device)
    ffdnet_path = os.path.join(BASE_DIR, 'models', 'ffdnet_model.pth')
    if os.path.exists(ffdnet_path):
        ckpt = torch.load(ffdnet_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            ffdnet.load_state_dict(ckpt['model_state_dict'])
        else:
            ffdnet.load_state_dict(ckpt)
        print("✓ FFDNet loaded")
    else:
        print(f"✗ FFDNet model not found at {ffdnet_path}!")
        print("Please train it using FFDnet.py")
        return

    # DnCNN
    dncnn = DnCNN(in_channels=1, depth=17, num_filters=64).to(device)
    dncnn_path = os.path.join(MODEL_DIR, 'dnCNN_model.pth')
    if not os.path.exists(dncnn_path):
        dncnn_path = os.path.join(MODEL_DIR, 'best_dncnn_checkpoint.pth') # Try alternative
        
    if os.path.exists(dncnn_path):
        ckpt = torch.load(dncnn_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            dncnn.load_state_dict(ckpt['model_state_dict'])
        else:
            dncnn.load_state_dict(ckpt)
        print(f"✓ DnCNN loaded ({dncnn_path})")
    else:
        print(f"✗ DnCNN model not found at {dncnn_path}!")
        print("Please train it using DnCNN.py")
        dncnn = None

    # 2. Load Test Image
    test_dir = os.path.join(BASE_DIR, 'datasets', 'Gray', 'test')
    if not os.path.exists(test_dir):
        print("Test directory not found.")
        return
        
    images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
    if not images:
        print("No images found.")
        return
        
    # Pick a nice image (or random)
    img_path = os.path.join(test_dir, images[0])
    print(f"Testing on: {images[0]}")
    
    clean_img = dataset.load_image(img_path)
    
    # 3. Run Tests
    sigma = 25
    noisy_img = dataset.add_gaussian_noise(clean_img, sigma)
    psnr_noisy = dataset.calculate_psnr(clean_img, noisy_img)
    
    # FFDNet Run
    print("Running FFDNet...")
    ffd_denoised, ffd_noise, ffd_time = run_ffdnet(ffdnet, noisy_img, sigma, device)
    ffd_psnr = dataset.calculate_psnr(clean_img, ffd_denoised)
    
    plot_single_model("FFDNet", clean_img, noisy_img, ffd_denoised, ffd_noise, psnr_noisy, ffd_psnr, ffd_time, sigma)
    
    # DnCNN Run
    if dncnn:
        print("Running DnCNN...")
        dn_denoised, dn_noise, dn_time = run_dncnn(dncnn, noisy_img, device)
        dn_psnr = dataset.calculate_psnr(clean_img, dn_denoised)
        
        plot_single_model("DnCNN", clean_img, noisy_img, dn_denoised, dn_noise, psnr_noisy, dn_psnr, dn_time, sigma)
        
        # Comparison
        plot_comparison(
            clean_img, noisy_img,
            {'img': ffd_denoised, 'time': ffd_time, 'psnr': ffd_psnr},
            {'img': dn_denoised, 'time': dn_time, 'psnr': dn_psnr},
            sigma
        )
    
    print("\nDone! Check the generated .png files.")

if __name__ == "__main__":
    main()
