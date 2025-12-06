import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from simple_unet import SimpleUNet

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, '..'))
GRAY_DATASET_DIR = os.path.join(ROOT_DIR, 'datasets', 'Gray')

def normalize_img(img):
    return img.astype(np.float32) / 255.0

def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma/255.0, img.shape).astype(np.float32)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    return noisy_img

def denoise_image(model, noisy_img, device):
    """Denoise a single image using the trained model"""
    model.eval()
    
    # Convert to tensor and add batch + channel dimensions
    img_tensor = torch.tensor(noisy_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        denoised = model(img_tensor)
    
    # Convert back to numpy
    denoised = denoised.squeeze().cpu().numpy()
    denoised = np.clip(denoised, 0.0, 1.0)
    
    return denoised

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = SimpleUNet(in_channels=1, out_channels=1, base_channels=64).to(device)
    model_path = os.path.join(CURR_DIR, 'n2v_unet_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Using random weights for demonstration")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    
    # Load a test image
    test_dir = os.path.join(GRAY_DATASET_DIR, 'test')
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not test_files:
        print("No test images found!")
        exit(1)
    
    TEST_FILE = test_files[20]

    test_img_path = os.path.join(test_dir, TEST_FILE)
    print(f"Testing on: {TEST_FILE}")
    
    # Load and process image
    img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    img = normalize_img(img)
    
    # Add noise
    sigma = 25
    noisy_img = add_gaussian_noise(img, sigma=sigma)
    
    # Denoise
    denoised_img = denoise_image(model, noisy_img, device)
    
    # Calculate PSNR
    mse_noisy = np.mean((img - noisy_img) ** 2)
    psnr_noisy = 10 * np.log10(1.0 / mse_noisy) if mse_noisy > 0 else float('inf')
    
    mse_denoised = np.mean((img - denoised_img) ** 2)
    psnr_denoised = 10 * np.log10(1.0 / mse_denoised) if mse_denoised > 0 else float('inf')
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original (Clean)')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Noisy (σ={sigma})\nPSNR: {psnr_noisy:.2f} dB')
    axes[1].axis('off')
    
    axes[2].imshow(denoised_img, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Denoised (N2V)\nPSNR: {psnr_denoised:.2f} dB')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults:")
    print(f"Noisy PSNR: {psnr_noisy:.2f} dB")
    print(f"Denoised PSNR: {psnr_denoised:.2f} dB")
    print(f"Improvement: {psnr_denoised - psnr_noisy:.2f} dB")
