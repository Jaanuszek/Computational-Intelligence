"""
Comparison script between FFDNet and DnCNN
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from includes import *
import torch.nn.functional as F_torch
from FFDnet import FFDNet, FFDNetConfig
from test_ffdnet import denoise_image as ffdnet_denoise

# Import DnCNN from parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class DnCNN(nn.Module):
    """DnCNN model for comparison"""
    def __init__(self, in_channels=1, depth=17, num_filters=64):
        super(DnCNN, self).__init__()
        layers = []
        
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(in_channels=num_filters, out_channels=in_channels, kernel_size=3, padding=1, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.dncnn(x)


def denoise_dncnn(model, image_path, device='cuda'):
    """Denoise using DnCNN"""
    if isinstance(device, str):
        device = torch.device(device)
    
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        predicted_noise = model(image_tensor)
        denoised_tensor = image_tensor - predicted_noise
        denoised_tensor = torch.clamp(denoised_tensor, 0, 1)
    
    denoised_image = denoised_tensor.squeeze().cpu().numpy()
    return denoised_image


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images (simplified version)"""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    mu1 = img1.mean()
    mu2 = img2.mean()
    
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
    
    return ssim


def compare_models(ffdnet_model, dncnn_model, image_path, sigma=25, device='cuda'):
    """Compare FFDNet and DnCNN on a single image"""
    if isinstance(device, str):
        device = torch.device(device)
    
    # Load original noisy image
    noisy_image = Image.open(image_path).convert('L')
    noisy_array = np.array(noisy_image).astype(np.float32) / 255.0
    
    # Denoise with FFDNet
    print("Denoising with FFDNet...")
    import time
    start = time.time()
    ffdnet_result = ffdnet_denoise(ffdnet_model, image_path, sigma, device)
    ffdnet_time = time.time() - start
    
    # Denoise with DnCNN
    print("Denoising with DnCNN...")
    start = time.time()
    dncnn_result = denoise_dncnn(dncnn_model, image_path, device)
    dncnn_time = time.time() - start
    
    # Try to load ground truth if available
    # Assuming ground truth might be in a parallel directory
    ground_truth = None
    try:
        # Try to find clean version
        if 'noisy_images' in image_path:
            gt_path = image_path.replace('noisy_images', 'ground_truth')
            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path).convert('L')
                ground_truth = np.array(gt_img).astype(np.float32) / 255.0
    except:
        pass
    
    # Calculate metrics if ground truth available
    if ground_truth is not None:
        # Resize results to match ground truth
        if ground_truth.shape != ffdnet_result.shape:
            from scipy.ndimage import zoom
            factors = (ground_truth.shape[0] / ffdnet_result.shape[0],
                      ground_truth.shape[1] / ffdnet_result.shape[1])
            ffdnet_result_resized = zoom(ffdnet_result, factors, order=1)
            dncnn_result_resized = zoom(dncnn_result, factors, order=1)
        else:
            ffdnet_result_resized = ffdnet_result
            dncnn_result_resized = dncnn_result
        
        ffdnet_psnr = calculate_psnr(ground_truth, ffdnet_result_resized)
        dncnn_psnr = calculate_psnr(ground_truth, dncnn_result_resized)
        
        ffdnet_ssim = calculate_ssim(ground_truth, ffdnet_result_resized)
        dncnn_ssim = calculate_ssim(ground_truth, dncnn_result_resized)
        
        print(f"\n{'='*60}")
        print(f"{'Metric':<20} {'FFDNet':<20} {'DnCNN':<20}")
        print(f"{'='*60}")
        print(f"{'PSNR (dB)':<20} {ffdnet_psnr:>19.4f} {dncnn_psnr:>19.4f}")
        print(f"{'SSIM':<20} {ffdnet_ssim:>19.4f} {dncnn_ssim:>19.4f}")
        print(f"{'Time (s)':<20} {ffdnet_time:>19.4f} {dncnn_time:>19.4f}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"{'Metric':<20} {'FFDNet':<20} {'DnCNN':<20}")
        print(f"{'='*60}")
        print(f"{'Time (s)':<20} {ffdnet_time:>19.4f} {dncnn_time:>19.4f}")
        print(f"{'='*60}")
        print("(Ground truth not available for PSNR/SSIM calculation)\n")
    
    # Visualize
    if ground_truth is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(noisy_array, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title(f'Noisy Image (σ={sigma})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(ffdnet_result, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title(f'FFDNet\nPSNR: {ffdnet_psnr:.2f} dB')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(dncnn_result, cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title(f'DnCNN\nPSNR: {dncnn_psnr:.2f} dB')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Ground Truth')
        axes[1, 0].axis('off')
        
        # Error maps
        ffdnet_error = np.abs(ground_truth - ffdnet_result_resized)
        dncnn_error = np.abs(ground_truth - dncnn_result_resized)
        
        axes[1, 1].imshow(ffdnet_error, cmap='hot', vmin=0, vmax=0.2)
        axes[1, 1].set_title('FFDNet Error Map')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(dncnn_error, cmap='hot', vmin=0, vmax=0.2)
        axes[1, 2].set_title('DnCNN Error Map')
        axes[1, 2].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(noisy_array, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Noisy Image (σ={sigma})')
        axes[0].axis('off')
        
        axes[1].imshow(ffdnet_result, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'FFDNet ({ffdnet_time:.3f}s)')
        axes[1].axis('off')
        
        axes[2].imshow(dncnn_result, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'DnCNN ({dncnn_time:.3f}s)')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('ffdnet_vs_dncnn_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return ffdnet_result, dncnn_result


def main():
    """Main comparison function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load FFDNet
    print("Loading FFDNet...")
    config = FFDNetConfig()
    ffdnet_model = FFDNet(
        in_channels=config.in_channels,
        num_features=config.num_features,
        num_conv_layers=config.num_conv_layers
    ).to(device)
    
    ffdnet_path = 'ffdnet_model.pth'
    if os.path.exists(ffdnet_path):
        ffdnet_model.load_state_dict(torch.load(ffdnet_path, map_location=device))
        print(f"✓ Loaded FFDNet from {ffdnet_path}")
    else:
        print(f"✗ FFDNet model not found at {ffdnet_path}")
        return
    
    # Load DnCNN
    print("Loading DnCNN...")
    dncnn_model = DnCNN(in_channels=1, depth=17, num_filters=64).to(device)
    
    dncnn_path = '../dnCNN_model.pth'
    if os.path.exists(dncnn_path):
        dncnn_model.load_state_dict(torch.load(dncnn_path, map_location=device))
        print(f"✓ Loaded DnCNN from {dncnn_path}")
    else:
        # Try alternative paths
        alt_paths = ['../best_dncnn_checkpoint.pth', '../../dnCNN_model.pth', '../../best_dncnn_checkpoint.pth']
        loaded = False
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                checkpoint = torch.load(alt_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    dncnn_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    dncnn_model.load_state_dict(checkpoint)
                print(f"✓ Loaded DnCNN from {alt_path}")
                loaded = True
                break
        
        if not loaded:
            print("✗ DnCNN model not found. Please train DnCNN first or specify correct path.")
            return
    
    # Find test image
    import dataset
    test_dir = dataset.NOISY_TEST_DIR
    
    if os.path.exists(test_dir):
        test_images = [f for f in os.listdir(test_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if test_images:
            test_image = os.path.join(test_dir, test_images[0])
            print(f"\nComparing models on: {test_image}\n")
            
            # Test with different sigma values
            for sigma in [15, 25, 50]:
                print(f"\n{'='*60}")
                print(f"Testing with sigma = {sigma}")
                print(f"{'='*60}\n")
                compare_models(ffdnet_model, dncnn_model, test_image, sigma, device)
        else:
            print("No test images found!")
    else:
        print(f"Test directory {test_dir} not found!")


if __name__ == "__main__":
    main()
