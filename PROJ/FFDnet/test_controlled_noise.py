"""
Test FFDNet with controlled noise - add noise ourselves to verify model works
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from includes import *
import torch.nn.functional as F_torch
from FFDnet import FFDNet, FFDNetConfig
from test_ffdnet import denoise_image
import dataset


def add_gaussian_noise(image, sigma):
    """Add Gaussian noise to image [0, 1]"""
    noise = np.random.normal(0, sigma / 255.0, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 1)


def calculate_psnr(img1, img2):
    """Calculate PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def test_with_controlled_noise(model, clean_path, sigma_noise=25, sigma_denoise=25, device='cuda'):
    """
    Test FFDNet with controlled noise level
    
    Args:
        model: trained model
        clean_path: path to clean image
        sigma_noise: noise level to add (0-75)
        sigma_denoise: sigma value for denoising (0-75)
        device: computation device
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    # Load clean image
    clean_image = Image.open(clean_path).convert('L')
    clean_array = np.array(clean_image).astype(np.float32) / 255.0
    
    # Add controlled Gaussian noise
    np.random.seed(42)  # For reproducibility
    noisy_array = add_gaussian_noise(clean_array, sigma_noise)
    
    # Save noisy image temporarily
    temp_noisy_path = 'temp_noisy.png'
    Image.fromarray((noisy_array * 255).astype(np.uint8), mode='L').save(temp_noisy_path)
    
    # Denoise
    denoised_array = denoise_image(model, temp_noisy_path, sigma_denoise, device)
    
    # Clean up
    os.remove(temp_noisy_path)
    
    # Calculate PSNR
    psnr_noisy = calculate_psnr(clean_array, noisy_array)
    psnr_denoised = calculate_psnr(clean_array, denoised_array)
    improvement = psnr_denoised - psnr_noisy
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(clean_array, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Clean (Ground Truth)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_array, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Noisy (σ={sigma_noise})\nPSNR: {psnr_noisy:.2f} dB', fontsize=14)
    axes[1].axis('off')
    
    color = 'green' if improvement > 1 else ('orange' if improvement > 0 else 'red')
    axes[2].imshow(denoised_array, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Denoised (σ_model={sigma_denoise})\nPSNR: {psnr_denoised:.2f} dB ({improvement:+.2f})',
                     fontsize=14, color=color, fontweight='bold')
    axes[2].axis('off')
    
    # Error map
    error = np.abs(clean_array - denoised_array)
    im = axes[3].imshow(error, cmap='hot', vmin=0, vmax=0.2)
    axes[3].set_title(f'Error Map\nMean Error: {error.mean():.4f}', fontsize=14)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(f'controlled_noise_test_sigma{sigma_noise}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return psnr_noisy, psnr_denoised, improvement


def comprehensive_test(model, clean_path, device='cuda'):
    """
    Test model with different noise and denoise sigma combinations
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE TEST: {os.path.basename(clean_path)}")
    print(f"{'='*80}\n")
    
    # Test different noise levels
    noise_levels = [15, 25, 35, 50]
    
    results = []
    
    for sigma_noise in noise_levels:
        print(f"Testing with noise level σ = {sigma_noise}...")
        
        # Test with different denoise sigmas
        best_psnr = -1
        best_sigma_denoise = 0
        
        for sigma_denoise in [15, 25, 35, 50]:
            psnr_noisy, psnr_denoised, improvement = test_with_controlled_noise(
                model, clean_path, sigma_noise, sigma_denoise, device
            )
            
            if psnr_denoised > best_psnr:
                best_psnr = psnr_denoised
                best_sigma_denoise = sigma_denoise
            
            results.append({
                'noise_sigma': sigma_noise,
                'denoise_sigma': sigma_denoise,
                'psnr_noisy': psnr_noisy,
                'psnr_denoised': psnr_denoised,
                'improvement': improvement
            })
        
        print(f"  Best result: σ_denoise={best_sigma_denoise}, PSNR={best_psnr:.2f} dB\n")
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'SUMMARY TABLE':.^80}")
    print(f"{'='*80}")
    print(f"{'Noise σ':<10} {'Denoise σ':<12} {'Noisy PSNR':<15} {'Denoised PSNR':<18} {'Improvement':<12}")
    print(f"{'-'*80}")
    
    for r in results:
        marker = "★" if r['improvement'] == max([x['improvement'] for x in results if x['noise_sigma'] == r['noise_sigma']]) else " "
        print(f"{marker} {r['noise_sigma']:<9} {r['denoise_sigma']:<11} {r['psnr_noisy']:<14.2f} {r['psnr_denoised']:<17.2f} {r['improvement']:>+10.2f} dB")
    
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    config = FFDNetConfig()
    model = FFDNet(
        in_channels=config.in_channels,
        num_features=config.num_features,
        num_conv_layers=config.num_conv_layers
    ).to(device)
    
    model_path = 'ffdnet_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Handle both checkpoint dict and direct state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from {model_path} (epoch {checkpoint.get('epoch', 'unknown')})\n")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ Loaded model from {model_path}\n")
    else:
        print(f"Model {model_path} not found!")
        sys.exit(1)
    
    model.eval()
    
    # Test on clean images with controlled noise
    clean_dir = dataset.GRAY_TEST_DIR
    clean_images = sorted([f for f in os.listdir(clean_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:2]
    
    for img_name in clean_images:
        clean_path = os.path.join(clean_dir, img_name)
        comprehensive_test(model, clean_path, device)
