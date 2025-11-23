import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os
from dnCNN import DnCNN


# class DnCNN(nn.Module):
#     """Minimal DnCNN to match training architecture (grayscale)
#     This matches the model defined in `dnCNN.py` used for training.
#     """
#     def __init__(self, in_channels=1, depth=17, num_filters=64):
#         super(DnCNN, self).__init__()
#         layers = []
#         layers.append(nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=3, padding=1, bias=False))
#         layers.append(nn.ReLU(inplace=True))
#         for _ in range(depth - 2):
#             layers.append(nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1, bias=False))
#             layers.append(nn.BatchNorm2d(num_filters))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Conv2d(in_channels=num_filters, out_channels=in_channels, kernel_size=3, padding=1, bias=False))
#         self.dncnn = nn.Sequential(*layers)

#     def forward(self, x):
#         # returns predicted noise
#         return self.dncnn(x)


def load_image_gray(path, resize=None):
    img = Image.open(path).convert('L')
    if resize is not None:
        # Pillow v9+ recommends Image.Resampling; fall back to Image.BILINEAR for older versions
        resample = getattr(Image, 'Resampling', None)
        if resample is not None:
            img = img.resize((resize, resize), resample.BILINEAR)
        else:
            img = img.resize((resize, resize), Image.BILINEAR)
    return img


def prepare_tensor(img_pil, device):
    to_tensor = transforms.ToTensor()
    t = to_tensor(img_pil).unsqueeze(0).to(device)  # shape (1,1,H,W)
    return t


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images (numpy arrays in range [0, 1])"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def visualize(before, after, ground_truth=None, out_path=None):
    """
    Visualize denoising results with PSNR metrics
    
    Args:
        before: noisy image (numpy array, shape (1, H, W))
        after: denoised image (numpy array, shape (1, H, W))
        ground_truth: optional clean image for PSNR calculation (numpy array, shape (1, H, W))
        out_path: optional path to save visualization
    """
    if ground_truth is not None:
        # Calculate PSNR
        psnr_noisy = calculate_psnr(ground_truth, before)
        psnr_denoised = calculate_psnr(ground_truth, after)
        improvement = psnr_denoised - psnr_noisy
        
        # Create 3-panel figure with ground truth
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        ax[0].imshow(ground_truth.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax[0].set_title('Ground Truth (Clean)', fontsize=12)
        ax[0].axis('off')

        ax[1].imshow(before.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax[1].set_title(f'Noisy\nPSNR: {psnr_noisy:.2f} dB', fontsize=12)
        ax[1].axis('off')
        
        ax[2].imshow(after.squeeze(), cmap='gray', vmin=0, vmax=1)
        color = 'green' if improvement > 0 else 'red'
        ax[2].set_title(f'Denoised (DnCNN)\nPSNR: {psnr_denoised:.2f} dB ({improvement:+.2f} dB)', 
                       fontsize=12, color=color, fontweight='bold')
        ax[2].axis('off')
        
        print(f"\n{'='*60}")
        print(f"PSNR Results:")
        print(f"{'='*60}")
        print(f"Noisy image PSNR:     {psnr_noisy:.2f} dB")
        print(f"Denoised image PSNR:  {psnr_denoised:.2f} dB")
        print(f"Improvement:          {improvement:+.2f} dB")
        print(f"{'='*60}\n")
    else:
        # No ground truth - just show before/after
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        ax[0].imshow(before.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax[0].set_title('Noisy', fontsize=12)
        ax[0].axis('off')

        ax[1].imshow(after.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax[1].set_title('Denoised (DnCNN)', fontsize=12, fontweight='bold')
        ax[1].axis('off')
        
        print("\nNote: No ground truth provided. To see PSNR metrics, provide a clean image path with --clean")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize denoising result using a saved DnCNN model')
    parser.add_argument('image', help='Path to noisy image (grayscale or RGB)')
    parser.add_argument('model', help='Path to saved model state_dict (pth)')
    parser.add_argument('--clean', default=None, help='Optional: path to clean/ground truth image for PSNR calculation')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to run the model on')
    parser.add_argument('--resize', type=int, default=None, help='Optional: resize the image to square size before denoising')
    parser.add_argument('--out', default=None, help='Optional: path to save the visualization PNG')
    args = parser.parse_args()

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    print(f"Working directory: {os.getcwd()}")
    print(f"Device: {device}")
    
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    img = load_image_gray(args.image, resize=args.resize)
    x = prepare_tensor(img, device)
    
    # Load ground truth if provided
    ground_truth_tensor = None
    if args.clean:
        if not os.path.exists(args.clean):
            print(f"Warning: Clean image not found: {args.clean}")
        else:
            clean_img = load_image_gray(args.clean, resize=args.resize)
            ground_truth_tensor = prepare_tensor(clean_img, device)

    model = DnCNN(in_channels=1)
    model.to(device)

    # Load state dict (supports dict with 'model_state_dict' or raw state_dict)
    ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    try:
        model.load_state_dict(state)
    except Exception as e:
        print('Failed to load state_dict into model:', e)
        raise

    model.eval()
    with torch.no_grad():
        pred_noise = model(x)
        denoised = x - pred_noise
        denoised = denoised.clamp(0.0, 1.0)

    # move to cpu numpy for visualization
    before_np = x.cpu().numpy()[0]
    after_np = denoised.cpu().numpy()[0]
    ground_truth_np = ground_truth_tensor.cpu().numpy()[0] if ground_truth_tensor is not None else None

    visualize(before_np, after_np, ground_truth=ground_truth_np, out_path=args.out)


if __name__ == '__main__':
    main()
