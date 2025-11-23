"""
Test script for FFDNet model - can be used for inference on individual images
"""
import sys
import os
from typing import Union
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from includes import *
import torch.nn.functional as F_torch
from FFDnet import FFDNet, FFDNetConfig

def denoise_image(model, image_path, sigma=25, device: Union[str, torch.device] = 'cuda'):
    """
    Denoise a single image using trained FFDNet model
    
    Args:
        model: trained FFDNet model
        image_path: path to the noisy image
        sigma: noise level (0-75)
        device: 'cuda' or 'cpu' or torch.device
    
    Returns:
        denoised_image: numpy array [H, W] with values in [0, 1]
    """
    # Ensure device is a torch.device
    if isinstance(device, str):
        device = torch.device(device)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 1, H, W]
    
    # Ensure dimensions are divisible by 2 (required for space-to-depth)
    _, _, h, w = image_tensor.shape
    new_h = h if h % 2 == 0 else h + 1
    new_w = w if w % 2 == 0 else w + 1
    
    if new_h != h or new_w != w:
        image_tensor = F_torch.pad(image_tensor, (0, new_w - w, 0, new_h - h), mode='reflect')
    
    # Create sigma map - same size as padded image
    sigma_map = torch.full((1, 1, new_h, new_w), sigma / 255.0, device=device)
    
    # Denoise
    model.eval()
    with torch.no_grad():
        predicted_noise = model(image_tensor, sigma_map)
        denoised_tensor = image_tensor - predicted_noise
        denoised_tensor = torch.clamp(denoised_tensor, 0, 1)
    
    # Crop back to original size and convert to numpy
    denoised_tensor = denoised_tensor[:, :, :h, :w]
    denoised_image = denoised_tensor.squeeze().cpu().numpy()
    
    return denoised_image


def visualize_denoising(model, image_path, sigma=25, device: Union[str, torch.device] = 'cuda'):
    """
    Visualize denoising result for a single image
    """
    # Ensure device is a torch.device
    if isinstance(device, str):
        device = torch.device(device)
    # Load original noisy image
    noisy_image = Image.open(image_path).convert('L')
    noisy_array = np.array(noisy_image).astype(np.float32) / 255.0
    
    # Denoise
    denoised_array = denoise_image(model, image_path, sigma, device)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(noisy_array, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Noisy Image (σ={sigma})')
    axes[0].axis('off')
    
    axes[1].imshow(denoised_array, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Denoised Image (FFDNet)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def batch_denoise(model, input_dir, output_dir, sigma=25, device: Union[str, torch.device] = 'cuda'):
    """
    Denoise all images in a directory
    
    Args:
        model: trained FFDNet model
        input_dir: directory containing noisy images
        output_dir: directory to save denoised images
        sigma: noise level
        device: 'cuda' or 'cpu' or torch.device
    """
    # Ensure device is a torch.device
    if isinstance(device, str):
        device = torch.device(device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"Denoising {len(image_files)} images...")
    
    for filename in tqdm(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            denoised = denoise_image(model, input_path, sigma, device)
            
            # Convert back to [0, 255] and save
            denoised_uint8 = (denoised * 255).astype(np.uint8)
            Image.fromarray(denoised_uint8, mode='L').save(output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"✓ Denoised images saved to {output_dir}")


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FFDNet denoising')
    parser.add_argument('--model', type=str, default='ffdnet_model.pth', 
                        help='Path to trained model')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image to denoise')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory of images to denoise')
    parser.add_argument('--output_dir', type=str, default='ffdnet_output',
                        help='Output directory for batch denoising')
    parser.add_argument('--sigma', type=int, default=25,
                        help='Noise level (0-75)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    config = FFDNetConfig()
    model = FFDNet(
        in_channels=config.in_channels,
        num_features=config.num_features,
        num_conv_layers=config.num_conv_layers
    ).to(device)
    
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"✓ Loaded model from {args.model}")
    else:
        print(f"Error: Model file {args.model} not found!")
        return
    
    # Denoise
    if args.image:
        print(f"Denoising single image: {args.image}")
        visualize_denoising(model, args.image, args.sigma, device)
    elif args.input_dir:
        print(f"Batch denoising from {args.input_dir} to {args.output_dir}")
        batch_denoise(model, args.input_dir, args.output_dir, args.sigma, device)
    else:
        print("Please provide --image or --input_dir")
        return


if __name__ == "__main__":
    # If no arguments provided, run a quick test
    if len(sys.argv) == 1:
        print("Quick test mode - loading model and testing on sample data")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load model
        config = FFDNetConfig()
        model = FFDNet(
            in_channels=config.in_channels,
            num_features=config.num_features,
            num_conv_layers=config.num_conv_layers
        ).to(device)
        
        model_path = 'ffdnet_model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Loaded model from {model_path}")
            
            # Test on a sample image from test set
            import dataset
            test_dir = dataset.NOISY_TEST_DIR
            
            if os.path.exists(test_dir):
                test_images = [f for f in os.listdir(test_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if test_images:
                    test_image = os.path.join(test_dir, test_images[0])
                    print(f"Testing on: {test_image}")
                    visualize_denoising(model, test_image, sigma=25, device=device)
                else:
                    print("No test images found!")
            else:
                print(f"Test directory {test_dir} not found!")
        else:
            print(f"Model file {model_path} not found! Train the model first.")
    else:
        main()
