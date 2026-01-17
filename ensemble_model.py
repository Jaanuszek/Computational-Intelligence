"""
Ensemble Denoising Model - Combining DnCNN, FFDNet, and Noise2Void

This module implements several ensemble strategies for combining the outputs
of three different denoising models: DnCNN, FFDNet, and Noise2Void.

Ensemble Methods:
1. Simple Average - Equal weighted average of all predictions
2. Weighted Average - Custom weights for each model
3. Adaptive Weighted - Weights based on local variance/confidence
4. Best of Three - Select best model prediction per region
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
from skimage.metrics import structural_similarity as ssim
import time

# Add paths for imports
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURR_DIR)
sys.path.append(os.path.join(CURR_DIR, 'FFDnet'))
sys.path.append(os.path.join(CURR_DIR, 'Noise2Void'))
sys.path.append(os.path.join(CURR_DIR, 'DnCNN'))

from DnCNN.dnCNN import DnCNN
from FFDnet import FFDNet
from simple_unet import SimpleUNet


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images (0-1 range)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images (0-1 range)"""
    return ssim(img1, img2, data_range=1.0)


class EnsembleDenoiser:
    """
    Ensemble denoising model combining DnCNN, FFDNet, and Noise2Void
    """
    
    def __init__(self, device='cuda', model_dir='models'):
        self.device = device
        self.model_dir = model_dir
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load all three models"""
        print("Loading ensemble models...")
        
        # Load DnCNN
        dncnn_path = os.path.join(CURR_DIR, 'models', 'best_dncnn_checkpoint.pth')
        if os.path.exists(dncnn_path):
            self.dncnn = DnCNN(depth=17, num_filters=64, in_channels=1).to(self.device)
            checkpoint = torch.load(dncnn_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.dncnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.dncnn.load_state_dict(checkpoint)
            self.dncnn.eval()
            print(f"✓ DnCNN loaded")
        else:
            raise FileNotFoundError(f"DnCNN model not found at {dncnn_path}")
        
        # Load FFDNet
        ffdnet_path = os.path.join(CURR_DIR, 'models', 'ffdnet_model.pth')
        if not os.path.exists(ffdnet_path):
            ffdnet_path = os.path.join(CURR_DIR, 'models', 'best_ffdnet_checkpoint.pth')
        
        if os.path.exists(ffdnet_path):
            self.ffdnet = FFDNet(in_channels=1, num_features=64, num_conv_layers=15).to(self.device)
            checkpoint = torch.load(ffdnet_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.ffdnet.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.ffdnet.load_state_dict(checkpoint)
            self.ffdnet.eval()
            print(f"✓ FFDNet loaded")
        else:
            raise FileNotFoundError(f"FFDNet model not found")
        
        # Load Noise2Void
        n2v_path = os.path.join(CURR_DIR, 'models', 'best_n2v_model.pth')
        if os.path.exists(n2v_path):
            self.n2v = SimpleUNet(in_channels=1, out_channels=1, base_channels=64).to(self.device)
            checkpoint = torch.load(n2v_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.n2v.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.n2v.load_state_dict(checkpoint)
            self.n2v.eval()
            print(f"✓ Noise2Void loaded")
        else:
            raise FileNotFoundError(f"Noise2Void model not found at {n2v_path}")
        
        print("All models loaded successfully!\n")
    
    def denoise_dncnn(self, noisy_img):
        """Denoise using DnCNN"""
        img_tensor = torch.tensor(noisy_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            noise = self.dncnn(img_tensor)
            denoised = img_tensor - noise
        denoised = denoised.squeeze().cpu().numpy()
        return np.clip(denoised, 0.0, 1.0)
    
    def denoise_ffdnet(self, noisy_img, sigma=25):
        """Denoise using FFDNet"""
        h, w = noisy_img.shape
        img_tensor = torch.tensor(noisy_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # FFDNet requires dimensions divisible by 2
        h_pad = (2 - h % 2) % 2
        w_pad = (2 - w % 2) % 2
        
        if h_pad > 0 or w_pad > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, w_pad, 0, h_pad), mode='reflect')
        
        # Create noise map
        sigma_map = torch.full_like(img_tensor, sigma / 255.0)
        
        with torch.no_grad():
            predicted_noise = self.ffdnet(img_tensor, sigma_map)
            denoised = img_tensor - predicted_noise
        
        # Crop back to original size
        if h_pad > 0 or w_pad > 0:
            denoised = denoised[:, :, :h, :w]
        
        denoised = denoised.squeeze().cpu().numpy()
        return np.clip(denoised, 0.0, 1.0)
    
    def denoise_n2v(self, noisy_img):
        """Denoise using Noise2Void"""
        img_tensor = torch.tensor(noisy_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            denoised = self.n2v(img_tensor)
        denoised = denoised.squeeze().cpu().numpy()
        return np.clip(denoised, 0.0, 1.0)
    
    def get_individual_predictions(self, noisy_img, sigma=25):
        """Get predictions from all three models"""
        pred_dncnn = self.denoise_dncnn(noisy_img)
        pred_ffdnet = self.denoise_ffdnet(noisy_img, sigma=sigma)
        pred_n2v = self.denoise_n2v(noisy_img)
        
        return {
            'DnCNN': pred_dncnn,
            'FFDNet': pred_ffdnet,
            'Noise2Void': pred_n2v
        }
    
    def ensemble_simple_average(self, noisy_img, sigma=25):
        """
        Simple ensemble: Equal weighted average of all three models
        """
        preds = self.get_individual_predictions(noisy_img, sigma)
        ensemble = (preds['DnCNN'] + preds['FFDNet'] + preds['Noise2Void']) / 3.0
        return np.clip(ensemble, 0.0, 1.0)
    
    def ensemble_weighted_average(self, noisy_img, sigma=25, weights=None):
        """
        Weighted ensemble: Custom weights for each model
        
        Args:
            weights: dict with keys 'DnCNN', 'FFDNet', 'Noise2Void'
                    If None, uses empirically determined weights
        """
        if weights is None:
            # Default weights - can be tuned based on validation performance
            # Assuming FFDNet and DnCNN perform slightly better (supervised learning)
            weights = {
                'DnCNN': 0.25,
                'FFDNet': 0.40,
                'Noise2Void': 0.35
            }
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        preds = self.get_individual_predictions(noisy_img, sigma)
        
        ensemble = (
            weights['DnCNN'] * preds['DnCNN'] +
            weights['FFDNet'] * preds['FFDNet'] +
            weights['Noise2Void'] * preds['Noise2Void']
        )
        
        return np.clip(ensemble, 0.0, 1.0)
    
    def ensemble_adaptive_weighted(self, noisy_img, sigma=25, window_size=8):
        """
        Adaptive weighted ensemble: Weights based on local image variance
        
        The idea: regions with high variance might benefit from different models
        - High variance regions: Give more weight to FFDNet (handles varying noise)
        - Low variance regions: More balanced weights
        """
        preds = self.get_individual_predictions(noisy_img, sigma)
        h, w = noisy_img.shape
        
        # Calculate local variance
        variance_map = cv2.blur(noisy_img ** 2, (window_size, window_size)) - \
                       cv2.blur(noisy_img, (window_size, window_size)) ** 2
        variance_map = np.clip(variance_map, 0, None)
        
        # Normalize variance to [0, 1]
        if variance_map.max() > 0:
            variance_map = variance_map / variance_map.max()
        
        # Create adaptive weights
        # High variance -> more FFDNet, less N2V
        # Low variance -> balanced
        w_ffdnet = 0.35 + 0.15 * variance_map
        w_dncnn = 0.35 - 0.05 * variance_map
        w_n2v = 1.0 - w_ffdnet - w_dncnn
        
        ensemble = (
            w_dncnn * preds['DnCNN'] +
            w_ffdnet * preds['FFDNet'] +
            w_n2v * preds['Noise2Void']
        )
        
        return np.clip(ensemble, 0.0, 1.0)
    
    def ensemble_median(self, noisy_img, sigma=25):
        """
        Median ensemble: Take median value across models
        More robust to outlier predictions
        """
        preds = self.get_individual_predictions(noisy_img, sigma)
        
        # Stack predictions
        pred_stack = np.stack([
            preds['DnCNN'],
            preds['FFDNet'],
            preds['Noise2Void']
        ], axis=0)
        
        # Take median
        ensemble = np.median(pred_stack, axis=0)
        return np.clip(ensemble, 0.0, 1.0)
    
    def denoise(self, noisy_img, sigma=25, method='weighted_average', **kwargs):
        """
        Main denoising function with ensemble
        
        Args:
            noisy_img: Noisy input image (numpy array, 0-1 range)
            sigma: Noise level (0-255 scale)
            method: Ensemble method - 'simple_average', 'weighted_average',
                   'adaptive_weighted', 'median', or individual model name
            **kwargs: Additional arguments for specific methods
        
        Returns:
            Denoised image (numpy array, 0-1 range)
        """
        if method == 'simple_average':
            return self.ensemble_simple_average(noisy_img, sigma)
        elif method == 'weighted_average':
            return self.ensemble_weighted_average(noisy_img, sigma, **kwargs)
        elif method == 'adaptive_weighted':
            return self.ensemble_adaptive_weighted(noisy_img, sigma, **kwargs)
        elif method == 'median':
            return self.ensemble_median(noisy_img, sigma)
        elif method == 'DnCNN':
            return self.denoise_dncnn(noisy_img)
        elif method == 'FFDNet':
            return self.denoise_ffdnet(noisy_img, sigma)
        elif method == 'Noise2Void':
            return self.denoise_n2v(noisy_img)
        else:
            raise ValueError(f"Unknown method: {method}")


def test_ensemble_on_image(image_path, sigma=25, device='cuda'):
    """
    Test all ensemble methods on a single image
    """
    import matplotlib.pyplot as plt
    
    # Load image
    if os.path.exists(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        clean_img = img.astype(np.float32) / 255.0
    else:
        print(f"Image not found: {image_path}")
        return
    
    # Add noise
    noise = np.random.normal(0, sigma/255.0, clean_img.shape).astype(np.float32)
    noisy_img = np.clip(clean_img + noise, 0.0, 1.0)
    
    # Initialize ensemble
    ensemble = EnsembleDenoiser(device=device)
    
    # Test all methods
    methods = [
        'DnCNN',
        'FFDNet', 
        'Noise2Void',
        'simple_average',
        'weighted_average',
        'adaptive_weighted',
        'median'
    ]
    
    results = {}
    print(f"\nTesting on image: {image_path}")
    print(f"Noise level (sigma): {sigma}")
    print(f"{'='*80}")
    
    # Noisy metrics
    psnr_noisy = calculate_psnr(clean_img, noisy_img)
    ssim_noisy = calculate_ssim(clean_img, noisy_img)
    print(f"Noisy Image - PSNR: {psnr_noisy:.2f} dB, SSIM: {ssim_noisy:.4f}")
    print(f"{'='*80}")
    
    for method in methods:
        start_time = time.time()
        denoised = ensemble.denoise(noisy_img, sigma=sigma, method=method)
        inference_time = time.time() - start_time
        
        psnr = calculate_psnr(clean_img, denoised)
        ssim_val = calculate_ssim(clean_img, denoised)
        
        results[method] = {
            'image': denoised,
            'psnr': psnr,
            'ssim': ssim_val,
            'time': inference_time
        }
        
        improvement_psnr = psnr - psnr_noisy
        improvement_ssim = ssim_val - ssim_noisy
        
        print(f"{method:20s} - PSNR: {psnr:6.2f} dB (+{improvement_psnr:5.2f}), "
              f"SSIM: {ssim_val:.4f} (+{improvement_ssim:+.4f}), "
              f"Time: {inference_time:.3f}s")
    
    # Visualize results
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Show original and noisy
    axes[0].imshow(clean_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Clean Image')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Noisy (σ={sigma})\nPSNR: {psnr_noisy:.2f} dB')
    axes[1].axis('off')
    
    # Show denoised results
    for idx, method in enumerate(methods, start=2):
        res = results[method]
        axes[idx].imshow(res['image'], cmap='gray', vmin=0, vmax=1)
        axes[idx].set_title(f'{method}\nPSNR: {res["psnr"]:.2f} dB, SSIM: {res["ssim"]:.4f}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CURR_DIR, 'ensemble_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: ensemble_comparison.png")
    plt.show()
    
    return results


def benchmark_ensemble(test_images_dir, sigma=25, device='cuda', num_images=10):
    """
    Comprehensive benchmark of all methods on multiple images
    """
    ensemble = EnsembleDenoiser(device=device)
    
    # Load test images
    test_images = []
    for root, dirs, files in os.walk(test_images_dir):
        for file in files[:num_images]:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    clean_img = img.astype(np.float32) / 255.0
                    test_images.append(clean_img)
    
    if len(test_images) == 0:
        print(f"No test images found in {test_images_dir}")
        return
    
    print(f"\nBenchmarking on {len(test_images)} images with σ={sigma}")
    print(f"{'='*100}")
    
    methods = [
        'DnCNN', 'FFDNet', 'Noise2Void',
        'simple_average', 'weighted_average', 'adaptive_weighted', 'median'
    ]
    
    # Initialize results storage
    results = {method: {'psnr': [], 'ssim': [], 'time': []} for method in methods}
    
    for idx, clean_img in enumerate(test_images):
        # Add noise
        noise = np.random.normal(0, sigma/255.0, clean_img.shape).astype(np.float32)
        noisy_img = np.clip(clean_img + noise, 0.0, 1.0)
        
        for method in methods:
            start_time = time.time()
            denoised = ensemble.denoise(noisy_img, sigma=sigma, method=method)
            inference_time = time.time() - start_time
            
            psnr = calculate_psnr(clean_img, denoised)
            ssim_val = calculate_ssim(clean_img, denoised)
            
            results[method]['psnr'].append(psnr)
            results[method]['ssim'].append(ssim_val)
            results[method]['time'].append(inference_time)
    
    # Print summary statistics
    print(f"\n{'Method':<25} {'Avg PSNR':<12} {'Avg SSIM':<12} {'Avg Time':<12}")
    print(f"{'='*100}")
    
    for method in methods:
        avg_psnr = np.mean(results[method]['psnr'])
        avg_ssim = np.mean(results[method]['ssim'])
        avg_time = np.mean(results[method]['time'])
        std_psnr = np.std(results[method]['psnr'])
        std_ssim = np.std(results[method]['ssim'])
        
        print(f"{method:<25} {avg_psnr:6.2f}±{std_psnr:4.2f} dB   "
              f"{avg_ssim:.4f}±{std_ssim:.4f}   {avg_time:6.3f}s")
    
    return results


if __name__ == "__main__":
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Test on a single image
    test_img_dir = os.path.join(CURR_DIR, 'datasets', 'Gray', 'test')
    if os.path.exists(test_img_dir):
        test_images = [f for f in os.listdir(test_img_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if test_images:
            rand_idx = np.random.randint(0, len(test_images))
            test_img_path = os.path.join(test_img_dir, test_images[rand_idx])
            test_ensemble_on_image(test_img_path, sigma=25, device=device)
            
            # Run full benchmark
            print("\n" + "="*100)
            print("Running full benchmark...")
            print("="*100)
            benchmark_ensemble(test_img_dir, sigma=25, device=device, num_images=10)
    else:
        print(f"Test directory not found: {test_img_dir}")
        print("Please update the path to your test images.")
