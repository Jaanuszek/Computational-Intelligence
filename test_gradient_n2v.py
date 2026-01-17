"""
Test and visualize Gradient-based N2V model
"""

import sys
sys.path.append('Noise2Void')

from simple_unet import SimpleUNet
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
import os

def load_images_from_dir(gt_dir, noisy_dir):
    """Load images from directories"""
    gt_images = []
    noisy_images = []
    filenames = []
    
    for filename in sorted(os.listdir(gt_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            gt_path = os.path.join(gt_dir, filename)
            noisy_path = os.path.join(noisy_dir, filename)
            
            if os.path.exists(noisy_path):
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
                
                if gt_img is not None and noisy_img is not None:
                    gt_images.append(gt_img)
                    noisy_images.append(noisy_img)
                    filenames.append(filename)
    
    return gt_images, noisy_images, filenames

def load_test_data():
    """Load test dataset from Gray and Noisy directories"""
    gt_dir = 'datasets/Gray/test'
    noisy_dir = 'datasets/Noisy/test'
    
    gt_images, noisy_images, filenames = load_images_from_dir(gt_dir, noisy_dir)
    print(f"Test set: {len(gt_images)} images")
    return gt_images, noisy_images, filenames

def denoise_image(model, noisy_img, device, patch_size=64, stride=32):
    """Denoise a full image by processing it in overlapping patches with blending"""
    model.eval()
    h, w = noisy_img.shape
    
    # Pad image to ensure full coverage
    pad_h = (patch_size - h % stride) % stride
    pad_w = (patch_size - w % stride) % stride
    
    if pad_h > 0 or pad_w > 0:
        noisy_padded = np.pad(noisy_img, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        noisy_padded = noisy_img
    
    h_pad, w_pad = noisy_padded.shape
    denoised = np.zeros_like(noisy_padded, dtype=np.float32)
    weights = np.zeros_like(noisy_padded, dtype=np.float32)
    
    # Create gaussian weight mask for blending
    y, x = np.ogrid[:patch_size, :patch_size]
    center_y, center_x = patch_size // 2, patch_size // 2
    gaussian_weight = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (patch_size / 4)**2))
    
    with torch.no_grad():
        # Process overlapping patches
        for i in range(0, h_pad - patch_size + 1, stride):
            for j in range(0, w_pad - patch_size + 1, stride):
                patch = noisy_padded[i:i+patch_size, j:j+patch_size]
                
                # Convert to tensor [1, 1, H, W]
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                
                # Denoise
                denoised_patch = model(patch_tensor)
                denoised_patch_np = denoised_patch.squeeze().cpu().numpy()
                
                # Accumulate with gaussian weighting
                denoised[i:i+patch_size, j:j+patch_size] += denoised_patch_np * gaussian_weight
                weights[i:i+patch_size, j:j+patch_size] += gaussian_weight
        
        # Handle edges if not fully covered
        if h_pad > patch_size:
            for j in range(0, w_pad - patch_size + 1, stride):
                i = h_pad - patch_size
                patch = noisy_padded[i:i+patch_size, j:j+patch_size]
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                denoised_patch = model(patch_tensor).squeeze().cpu().numpy()
                denoised[i:i+patch_size, j:j+patch_size] += denoised_patch * gaussian_weight
                weights[i:i+patch_size, j:j+patch_size] += gaussian_weight
        
        if w_pad > patch_size:
            for i in range(0, h_pad - patch_size + 1, stride):
                j = w_pad - patch_size
                patch = noisy_padded[i:i+patch_size, j:j+patch_size]
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                denoised_patch = model(patch_tensor).squeeze().cpu().numpy()
                denoised[i:i+patch_size, j:j+patch_size] += denoised_patch * gaussian_weight
                weights[i:i+patch_size, j:j+patch_size] += gaussian_weight
    
    # Normalize by weights
    denoised = np.divide(denoised, weights, where=weights > 0)
    
    # Remove padding
    denoised = denoised[:h, :w]
    
    return denoised

def evaluate_model(model, gt_images, noisy_images, device):
    """Evaluate model on test set"""
    psnr_values = []
    ssim_values = []
    
    print(f"Evaluating on {len(noisy_images)} images...")
    for i, (gt_img, noisy_img) in enumerate(zip(gt_images, noisy_images)):
        # Denoise
        denoised_np = denoise_image(model, noisy_img, device)
        
        # Clip to valid range
        denoised_np = np.clip(denoised_np, 0, 255)
        gt_img = np.clip(gt_img, 0, 255)
        
        # Calculate metrics
        psnr_val = psnr(gt_img, denoised_np, data_range=255)
        ssim_val = ssim(gt_img, denoised_np, data_range=255)
        
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(noisy_images)} images...")
    
    return np.mean(psnr_values), np.mean(ssim_values), psnr_values, ssim_values

def visualize_results(model, gt_images, noisy_images, filenames, device, num_examples=6):
    """Visualize denoising results"""
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 2.5*num_examples))
    
    # Select random indices
    indices = np.random.choice(len(noisy_images), min(num_examples, len(noisy_images)), replace=False)
    
    for idx, i in enumerate(indices):
        gt_img = gt_images[i]
        noisy_img = noisy_images[i]
        
        # Denoise
        denoised_np = denoise_image(model, noisy_img, device)
        
        # Clip
        denoised_np = np.clip(denoised_np, 0, 255)
        gt_img = np.clip(gt_img, 0, 255)
        noisy_img = np.clip(noisy_img, 0, 255)
        
        # Calculate metrics
        psnr_noisy = psnr(gt_img, noisy_img, data_range=255)
        psnr_denoised = psnr(gt_img, denoised_np, data_range=255)
        ssim_denoised = ssim(gt_img, denoised_np, data_range=255)
        
        # Plot
        axes[idx, 0].imshow(noisy_img, cmap='gray', vmin=0, vmax=255)
        axes[idx, 0].set_title(f'Noisy - {filenames[i]}\nPSNR: {psnr_noisy:.2f} dB')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(denoised_np, cmap='gray', vmin=0, vmax=255)
        axes[idx, 1].set_title(f'Denoised (Gradient N2V)\nPSNR: {psnr_denoised:.2f} dB\nSSIM: {ssim_denoised:.4f}')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(gt_img, cmap='gray', vmin=0, vmax=255)
        axes[idx, 2].set_title('Ground Truth')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradient_n2v_results.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to gradient_n2v_results.png")
    plt.show()

def main():
    print("="*60)
    print("TESTING GRADIENT-BASED N2V MODEL")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    model_path = 'Noise2Void/n2v_unet_model_gradient.pth'
    print(f"\nLoading model from {model_path}...")
    model = SimpleUNet(in_channels=1, out_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully!")
    
    # Load test data
    print("\nLoading test data...")
    gt_images, noisy_images, filenames = load_test_data()
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    avg_psnr, avg_ssim, psnr_vals, ssim_vals = evaluate_model(model, gt_images, noisy_images, device)
    
    print(f"\nResults on {len(psnr_vals)} test images:")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    print(f"  PSNR std: {np.std(psnr_vals):.2f} dB")
    print(f"  SSIM std: {np.std(ssim_vals):.4f}")
    
    # Calculate noisy baseline
    print("\nCalculating noisy baseline...")
    noisy_psnr = []
    for gt_img, noisy_img in zip(gt_images, noisy_images):
        noisy_psnr.append(psnr(gt_img, noisy_img, data_range=255))
    
    print(f"  Noisy PSNR: {np.mean(noisy_psnr):.2f} dB")
    print(f"  Improvement: +{avg_psnr - np.mean(noisy_psnr):.2f} dB")
    
    # Visualize
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    visualize_results(model, gt_images, noisy_images, filenames, device, num_examples=6)
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
