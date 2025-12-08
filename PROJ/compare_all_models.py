import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import sys
from skimage.metrics import structural_similarity as ssim

# Add paths for imports
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURR_DIR)
sys.path.append(os.path.join(CURR_DIR, 'FFDnet'))
sys.path.append(os.path.join(CURR_DIR, 'Noise2Void'))

from dnCNN import DnCNN
from FFDnet import FFDNet
from Noise2Void.simple_unet import SimpleUNet

ROOT_DIR = CURR_DIR
GRAY_DATASET_DIR = os.path.join(ROOT_DIR, 'datasets', 'Gray')

def normalize_img(img):
    return img.astype(np.float32) / 255.0

def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma/255.0, img.shape).astype(np.float32)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    return noisy_img

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=1.0)

def denoise_dncnn(model, noisy_img, device):
    model.eval()
    img_tensor = torch.tensor(noisy_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        noise = model(img_tensor)
        denoised = img_tensor - noise
    denoised = denoised.squeeze().cpu().numpy()
    return np.clip(denoised, 0.0, 1.0)

def denoise_ffdnet(model, noisy_img, device, sigma=25):
    model.eval()
    h, w = noisy_img.shape
    img_tensor = torch.tensor(noisy_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # FFDNet requires dimensions divisible by 2 (due to PixelUnshuffle)
    h_pad = (2 - h % 2) % 2
    w_pad = (2 - w % 2) % 2
    
    if h_pad > 0 or w_pad > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, w_pad, 0, h_pad), mode='reflect')
    
    # Create noise map
    sigma_map = torch.full_like(img_tensor, sigma / 255.0)
    
    with torch.no_grad():
        predicted_noise = model(img_tensor, sigma_map)
        denoised = img_tensor - predicted_noise  # Residual learning
    
    # Crop back to original size if padded
    if h_pad > 0 or w_pad > 0:
        denoised = denoised[:, :, :h, :w]
    
    denoised = denoised.squeeze().cpu().numpy()
    return np.clip(denoised, 0.0, 1.0)

def denoise_n2v(model, noisy_img, device):
    model.eval()
    img_tensor = torch.tensor(noisy_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised = model(img_tensor)
    denoised = denoised.squeeze().cpu().numpy()
    return np.clip(denoised, 0.0, 1.0)

def get_model_size(model_path):
    """Get model size in MB"""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    return 0

def load_models(device):
    """Load all three models"""
    models = {}
    
    # DnCNN
    dncnn_path = os.path.join(CURR_DIR, 'best_dncnn_checkpoint.pth')
    if os.path.exists(dncnn_path):
        dncnn = DnCNN(depth=17, num_filters=64, in_channels=1).to(device)
        checkpoint = torch.load(dncnn_path, map_location=device, weights_only=False)
        # Check if checkpoint is a dict with 'model_state_dict' key
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            dncnn.load_state_dict(checkpoint['model_state_dict'])
        else:
            dncnn.load_state_dict(checkpoint)
        models['DnCNN'] = {
            'model': dncnn,
            'denoise_fn': denoise_dncnn,
            'path': dncnn_path,
            'size_mb': get_model_size(dncnn_path)
        }
        print(f"✓ DnCNN loaded from {dncnn_path}")
    else:
        print(f"✗ DnCNN not found at {dncnn_path}")
    
    # FFDNet - try final model first, then checkpoint
    ffdnet_path = os.path.join(CURR_DIR, 'FFDnet', 'ffdnet_model.pth')
    if not os.path.exists(ffdnet_path):
        ffdnet_path = os.path.join(CURR_DIR, 'FFDnet', 'best_ffdnet_checkpoint.pth')
    
    if os.path.exists(ffdnet_path):
        ffdnet = FFDNet(in_channels=1, num_features=64, num_conv_layers=15).to(device)
        checkpoint = torch.load(ffdnet_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            ffdnet.load_state_dict(checkpoint['model_state_dict'])
        else:
            ffdnet.load_state_dict(checkpoint)
        models['FFDNet'] = {
            'model': ffdnet,
            'denoise_fn': denoise_ffdnet,
            'path': ffdnet_path,
            'size_mb': get_model_size(ffdnet_path)
        }
        print(f"✓ FFDNet loaded from {ffdnet_path}")
    else:
        print(f"✗ FFDNet not found")
    
    # Noise2Void
    n2v_path = os.path.join(CURR_DIR, 'Noise2Void', 'best_n2v_model.pth')
    if os.path.exists(n2v_path):
        n2v = SimpleUNet(in_channels=1, out_channels=1, base_channels=64).to(device)
        checkpoint = torch.load(n2v_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            n2v.load_state_dict(checkpoint['model_state_dict'])
        else:
            n2v.load_state_dict(checkpoint)
        models['Noise2Void'] = {
            'model': n2v,
            'denoise_fn': denoise_n2v,
            'path': n2v_path,
            'size_mb': get_model_size(n2v_path)
        }
        print(f"✓ Noise2Void loaded from {n2v_path}")
    else:
        print(f"✗ Noise2Void not found at {n2v_path}")
    
    return models

def benchmark_models(models, test_images, sigma=25, device='cuda'):
    """Benchmark all models on test images"""
    results = {name: {
        'psnr_improvements': [],
        'ssim_improvements': [],
        'inference_times': [],
        'psnr_noisy': [],
        'psnr_denoised': [],
        'ssim_noisy': [],
        'ssim_denoised': []
    } for name in models.keys()}
    
    print(f"\n{'='*60}")
    print(f"Testing on {len(test_images)} images with sigma={sigma}")
    print(f"{'='*60}\n")
    
    for idx, clean_img in enumerate(test_images):
        print(f"Processing image {idx+1}/{len(test_images)}...")
        
        # Add noise
        noisy_img = add_gaussian_noise(clean_img, sigma=sigma)
        
        # Calculate noisy metrics
        psnr_noisy = calculate_psnr(clean_img, noisy_img)
        ssim_noisy = calculate_ssim(clean_img, noisy_img)
        
        for model_name, model_data in models.items():
            model = model_data['model']
            denoise_fn = model_data['denoise_fn']
            
            # Measure inference time
            start_time = time.time()
            
            if model_name == 'FFDNet':
                denoised_img = denoise_fn(model, noisy_img, device, sigma=sigma)
            else:
                denoised_img = denoise_fn(model, noisy_img, device)
            
            inference_time = time.time() - start_time
            
            # Calculate denoised metrics
            psnr_denoised = calculate_psnr(clean_img, denoised_img)
            ssim_denoised = calculate_ssim(clean_img, denoised_img)
            
            # Store results
            results[model_name]['psnr_noisy'].append(psnr_noisy)
            results[model_name]['psnr_denoised'].append(psnr_denoised)
            results[model_name]['psnr_improvements'].append(psnr_denoised - psnr_noisy)
            results[model_name]['ssim_noisy'].append(ssim_noisy)
            results[model_name]['ssim_denoised'].append(ssim_denoised)
            results[model_name]['ssim_improvements'].append(ssim_denoised - ssim_noisy)
            results[model_name]['inference_times'].append(inference_time)
            
            print(f"  {model_name:12s}: PSNR {psnr_noisy:.2f} → {psnr_denoised:.2f} dB (+{psnr_denoised-psnr_noisy:.2f}), "
                  f"SSIM {ssim_noisy:.4f} → {ssim_denoised:.4f} (+{ssim_denoised-ssim_noisy:.4f}), "
                  f"Time: {inference_time*1000:.1f}ms")
    
    return results

def print_summary(models, results):
    """Print comparison summary"""
    print(f"\n{'='*80}")
    print(f"{'SUMMARY - Model Comparison':^80}")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Model':<15} {'PSNR Gain':<15} {'SSIM Gain':<15} {'Avg Time':<15} {'Size (MB)':<12}")
    print(f"{'-'*80}")
    
    for model_name, model_data in models.items():
        avg_psnr = np.mean(results[model_name]['psnr_improvements'])
        avg_ssim = np.mean(results[model_name]['ssim_improvements'])
        avg_time = np.mean(results[model_name]['inference_times']) * 1000  # ms
        size_mb = model_data['size_mb']
        
        print(f"{model_name:<15} {avg_psnr:>6.2f} dB      {avg_ssim:>6.4f}        "
              f"{avg_time:>7.1f} ms      {size_mb:>7.2f} MB")
    
    print(f"{'='*80}\n")
    
    # Find best models
    best_psnr = max(models.keys(), key=lambda x: np.mean(results[x]['psnr_improvements']))
    best_ssim = max(models.keys(), key=lambda x: np.mean(results[x]['ssim_improvements']))
    best_speed = min(models.keys(), key=lambda x: np.mean(results[x]['inference_times']))
    best_size = min(models.keys(), key=lambda x: models[x]['size_mb'])
    
    print(f"🏆 Best PSNR Improvement: {best_psnr} ({np.mean(results[best_psnr]['psnr_improvements']):.2f} dB)")
    print(f"🏆 Best SSIM Improvement: {best_ssim} ({np.mean(results[best_ssim]['ssim_improvements']):.4f})")
    print(f"⚡ Fastest Inference:     {best_speed} ({np.mean(results[best_speed]['inference_times'])*1000:.1f} ms)")
    print(f"💾 Smallest Model:        {best_size} ({models[best_size]['size_mb']:.2f} MB)")
    print()

def plot_comparison(models, results):
    """Create comparison plots"""
    model_names = list(models.keys())
    n_models = len(model_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. PSNR Improvement
    ax = axes[0, 0]
    psnr_gains = [np.mean(results[name]['psnr_improvements']) for name in model_names]
    psnr_stds = [np.std(results[name]['psnr_improvements']) for name in model_names]
    bars = ax.bar(model_names, psnr_gains, yerr=psnr_stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('PSNR Improvement (dB)', fontweight='bold')
    ax.set_title('Average PSNR Gain')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, psnr_gains)):
        ax.text(bar.get_x() + bar.get_width()/2, val + psnr_stds[i] + 0.1, f'{val:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # 2. SSIM Improvement
    ax = axes[0, 1]
    ssim_gains = [np.mean(results[name]['ssim_improvements']) for name in model_names]
    ssim_stds = [np.std(results[name]['ssim_improvements']) for name in model_names]
    bars = ax.bar(model_names, ssim_gains, yerr=ssim_stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('SSIM Improvement', fontweight='bold')
    ax.set_title('Average SSIM Gain')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, ssim_gains)):
        ax.text(bar.get_x() + bar.get_width()/2, val + ssim_stds[i] + 0.001, f'{val:.4f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Inference Time
    ax = axes[1, 0]
    times = [np.mean(results[name]['inference_times']) * 1000 for name in model_names]
    time_stds = [np.std(results[name]['inference_times']) * 1000 for name in model_names]
    bars = ax.bar(model_names, times, yerr=time_stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Inference Time (ms)', fontweight='bold')
    ax.set_title('Average Inference Speed')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, times)):
        ax.text(bar.get_x() + bar.get_width()/2, val + time_stds[i] + 0.5, f'{val:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # 4. Model Size
    ax = axes[1, 1]
    sizes = [models[name]['size_mb'] for name in model_names]
    bars = ax.bar(model_names, sizes, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Model Size (MB)', fontweight='bold')
    ax.set_title('Model File Size on Disk')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(CURR_DIR, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {output_path}")
    
    plt.show()

def plot_visual_comparison(models, test_images, sigma=25, device='cuda'):
    """Create visual comparison of denoising results"""
    # Use first 3 test images
    n_samples = min(3, len(test_images))
    
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Visual Comparison of Denoising Methods', fontsize=16, fontweight='bold', y=0.995)
    
    model_names = list(models.keys())
    
    for img_idx in range(n_samples):
        clean_img = test_images[img_idx]
        noisy_img = add_gaussian_noise(clean_img, sigma=sigma)
        
        # Original
        axes[img_idx, 0].imshow(clean_img, cmap='gray', vmin=0, vmax=1)
        axes[img_idx, 0].set_title('Original (Clean)', fontweight='bold')
        axes[img_idx, 0].axis('off')
        
        # Noisy
        psnr_noisy = calculate_psnr(clean_img, noisy_img)
        ssim_noisy = calculate_ssim(clean_img, noisy_img)
        axes[img_idx, 1].imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
        axes[img_idx, 1].set_title(f'Noisy (σ={sigma})\nPSNR: {psnr_noisy:.2f} dB\nSSIM: {ssim_noisy:.4f}', 
                                    fontweight='bold')
        axes[img_idx, 1].axis('off')
        
        # Denoise with all models and store results
        denoised_results = {}
        for model_name in model_names:
            model_data = models[model_name]
            model = model_data['model']
            denoise_fn = model_data['denoise_fn']
            
            if model_name == 'FFDNet':
                denoised_img = denoise_fn(model, noisy_img, device, sigma=sigma)
            else:
                denoised_img = denoise_fn(model, noisy_img, device)
            
            psnr_denoised = calculate_psnr(clean_img, denoised_img)
            ssim_denoised = calculate_ssim(clean_img, denoised_img)
            psnr_gain = psnr_denoised - psnr_noisy
            
            denoised_results[model_name] = {
                'image': denoised_img,
                'psnr': psnr_denoised,
                'ssim': ssim_denoised,
                'psnr_gain': psnr_gain,
                'ssim_gain': ssim_denoised - ssim_noisy
            }
        
        # Find best PSNR gain
        best_psnr_gain = max([res['psnr_gain'] for res in denoised_results.values()])
        
        # Display denoised images
        for model_idx, model_name in enumerate(model_names):
            result = denoised_results[model_name]
            
            axes[img_idx, 2 + model_idx].imshow(result['image'], cmap='gray', vmin=0, vmax=1)
            
            # Highlight best result in green
            is_best = (result['psnr_gain'] == best_psnr_gain)
            title_color = 'green' if is_best else 'black'
            
            axes[img_idx, 2 + model_idx].set_title(
                f'{model_name}\nPSNR: {result["psnr"]:.2f} dB (+{result["psnr_gain"]:.2f})\nSSIM: {result["ssim"]:.4f} (+{result["ssim_gain"]:.4f})',
                fontweight='bold',
                color=title_color
            )
            axes[img_idx, 2 + model_idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(CURR_DIR, 'visual_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📷 Visual comparison saved to: {output_path}")
    
    plt.show()

def plot_noise_robustness(models, test_images, device='cuda'):
    """Test models with different noise levels"""
    sigma_levels = [10, 25, 50, 75]
    model_names = list(models.keys())
    
    print(f"\n{'='*80}")
    print(f"{'NOISE ROBUSTNESS TEST':^80}")
    print(f"{'='*80}\n")
    
    # Store results for each sigma
    results_by_sigma = {}
    
    for sigma in sigma_levels:
        print(f"\nTesting with σ={sigma}...")
        results = benchmark_models(models, test_images, sigma=sigma, device=device)
        results_by_sigma[sigma] = results
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Performance vs Noise Level', fontsize=16, fontweight='bold')
    
    # PSNR improvement vs sigma
    ax = axes[0]
    for model_name in model_names:
        psnr_gains = [np.mean(results_by_sigma[sigma][model_name]['psnr_improvements']) 
                      for sigma in sigma_levels]
        ax.plot(sigma_levels, psnr_gains, marker='o', linewidth=2, label=model_name, markersize=8)
    
    ax.set_xlabel('Noise Level (σ)', fontweight='bold', fontsize=12)
    ax.set_ylabel('PSNR Improvement (dB)', fontweight='bold', fontsize=12)
    ax.set_title('PSNR Gain vs Noise Level', fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sigma_levels)
    
    # SSIM improvement vs sigma
    ax = axes[1]
    for model_name in model_names:
        ssim_gains = [np.mean(results_by_sigma[sigma][model_name]['ssim_improvements']) 
                      for sigma in sigma_levels]
        ax.plot(sigma_levels, ssim_gains, marker='s', linewidth=2, label=model_name, markersize=8)
    
    ax.set_xlabel('Noise Level (σ)', fontweight='bold', fontsize=12)
    ax.set_ylabel('SSIM Improvement', fontweight='bold', fontsize=12)
    ax.set_title('SSIM Gain vs Noise Level', fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sigma_levels)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(CURR_DIR, 'noise_robustness.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Noise robustness plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'NOISE ROBUSTNESS SUMMARY':^80}")
    print(f"{'='*80}\n")
    
    for model_name in model_names:
        print(f"\n{model_name}:")
        print(f"{'Sigma':<10} {'PSNR Gain':<15} {'SSIM Gain':<15}")
        print(f"{'-'*40}")
        for sigma in sigma_levels:
            psnr = np.mean(results_by_sigma[sigma][model_name]['psnr_improvements'])
            ssim = np.mean(results_by_sigma[sigma][model_name]['ssim_improvements'])
            print(f"{sigma:<10} {psnr:>6.2f} dB      {ssim:>6.4f}")
    print()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load models
    models = load_models(device)
    
    if len(models) == 0:
        print("❌ No models found! Please train the models first.")
        return
    
    # Load test images
    test_dir = os.path.join(GRAY_DATASET_DIR, 'test')
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(test_files) == 0:
        print(f"❌ No test images found in {test_dir}")
        return
    
    # Load first 10 test images (or all if less than 10)
    n_test = min(10, len(test_files))
    test_images = []
    for i in range(n_test):
        img_path = os.path.join(test_dir, test_files[i])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = normalize_img(img)
        test_images.append(img)
    
    # Benchmark models
    results = benchmark_models(models, test_images, sigma=25, device=device)
    
    # Print summary
    print_summary(models, results)
    
    # Plot comparison
    plot_comparison(models, results)
    
    # Plot visual comparison (first 3 images)
    plot_visual_comparison(models, test_images, sigma=25, device=device)
    
    # Test noise robustness
    plot_noise_robustness(models, test_images, device=device)

if __name__ == "__main__":
    main()
