import sys
import os
from typing import Union
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from includes import *
import torch.nn.functional as F_torch
from FFDnet import FFDNet, FFDNetConfig


def ffdnet_denoise_tiled(model, noisy_tensor, sigma, patch=64, overlap=16):
    """
    Perform tiled denoising (patch-based) — REQUIRED for FFDNet trained on patches.
    noisy_tensor: [1,1,H,W] in [0,1]
    sigma: scalar noise level (0–75)
    """
    model.eval()
    _, _, H, W = noisy_tensor.shape

    denoised = torch.zeros_like(noisy_tensor)
    weight = torch.zeros_like(noisy_tensor)

    sigma_map = torch.full_like(noisy_tensor, sigma/255.0)

    step = patch - overlap

    # Calculate grid positions to ensure full coverage
    y_positions = list(range(0, H - patch + 1, step))
    if not y_positions or y_positions[-1] + patch < H:
        y_positions.append(H - patch)
        
    x_positions = list(range(0, W - patch + 1, step))
    if not x_positions or x_positions[-1] + patch < W:
        x_positions.append(W - patch)

    for y in y_positions:
        for x in x_positions:
            
            noisy_patch = noisy_tensor[:, :, y:y+patch, x:x+patch]
            sigma_patch = sigma_map[:, :, y:y+patch, x:x+patch]

            with torch.no_grad():
                predicted_noise = model(noisy_patch, sigma_patch)
                # print("predicted_noise mean:", predicted_noise.abs().mean().item())
                clean_patch = noisy_patch - predicted_noise

            denoised[:, :, y:y+patch, x:x+patch] += clean_patch
            weight[:, :, y:y+patch, x:x+patch] += 1

    # avoid division by zero (edges)
    denoised /= torch.clamp(weight, min=1)

    return denoised

def denoise_image(model, image_path, sigma=25, device='cuda'):
    """
    FFDNet inference on full image (no tiling).
    Handles padding if dimensions are not divisible by 2.
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Load grayscale image
    img = Image.open(image_path).convert('L')
    img_np = np.array(img).astype(np.float32) / 255.0

    # Convert to torch tensor
    noisy_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    
    # FFDNet requires dimensions divisible by 2 (due to PixelUnshuffle)
    _, _, H, W = noisy_tensor.shape
    h_pad = (2 - H % 2) % 2
    w_pad = (2 - W % 2) % 2
    
    if h_pad > 0 or w_pad > 0:
        # Pad with reflection to avoid boundary artifacts
        noisy_tensor = F_torch.pad(noisy_tensor, (0, w_pad, 0, h_pad), mode='reflect')

    # Create sigma map
    sigma_map = torch.full_like(noisy_tensor, sigma/255.0)

    # Run inference
    model.eval()
    with torch.no_grad():
        predicted_noise = model(noisy_tensor, sigma_map)
        denoised_tensor = noisy_tensor - predicted_noise

    # Crop back to original size if padded
    if h_pad > 0 or w_pad > 0:
        denoised_tensor = denoised_tensor[:, :, :H, :W]

    return denoised_tensor.squeeze().cpu().numpy()

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def visualize_denoising(model, image_path, sigma=25, device='cuda'):
    device = torch.device(device)
    noisy = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
    denoised = denoise_image(model, image_path, sigma, device)

    # Try to find ground truth
    ground_truth = None
    psnr_noisy = None
    psnr_val = None
    improvement = None
    
    try:
        import dataset
        filename = os.path.basename(image_path)
        # Check standard locations
        possible_gt_dirs = [
            dataset.ORIGINAL_TEST_GROUND_TRUTH_DIR,
            dataset.ORIGINAL_VALIDATE_GROUND_TRUTH_DIR,
            dataset.ORIGINAL_TRAIN_GROUND_TRUTH_DIR
        ]
        
        for gt_dir in possible_gt_dirs:
            gt_path = os.path.join(gt_dir, filename)
            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path).convert('L')
                gt_np = np.array(gt_img).astype(np.float32) / 255.0
                
                if gt_np.shape == denoised.shape:
                    ground_truth = gt_np
                    psnr_val = calculate_psnr(ground_truth, denoised)
                    psnr_noisy = calculate_psnr(ground_truth, noisy)
                    improvement = psnr_val - psnr_noisy
                    print(f"Found ground truth: {gt_path}")
                    print(f"PSNR: {psnr_val:.2f} dB")
                    break
    except Exception as e:
        print(f"Could not load ground truth: {e}")

    if ground_truth is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Noisy (σ={sigma})')
        axes[1].axis('off')

        color = 'black'
        if improvement is not None:
            color = 'green' if improvement > 0 else 'red'        
        axes[2].imshow(denoised, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'Denoised (FFDNet)\nPSNR: {psnr_val:.2f} dB ({improvement:+.2f} dB)',
                          fontsize = 12, color = color, fontweight='bold')
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(noisy, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Noisy (σ={sigma})')
        axes[0].axis('off')

    plt.tight_layout()
    plt.show()

def batch_denoise(model, input_dir, output_dir, sigma=25, device='cuda'):
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    for fn in tqdm(os.listdir(input_dir)):
        if not fn.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
            continue

        inp = os.path.join(input_dir, fn)
        out = os.path.join(output_dir, fn)

        try:
            den = denoise_image(model, inp, sigma, device)
            Image.fromarray((den*255).astype(np.uint8), 'L').save(out)
        except Exception as e:
            print(f"Error processing {fn}: {e}")

    print("✓ Batch denoising completed.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ffdnet_model.pth')
    parser.add_argument('--image', type=str)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='ffdnet_output')
    parser.add_argument('--sigma', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    config = FFDNetConfig()
    model = FFDNet(config.in_channels, config.num_features, config.num_conv_layers).to(device)
    ffdnet_path = os.path.join(MODEL_DIR, 'ffdnet_model.pth')
    if os.path.exists(ffdnet_path):
        model.load_state_dict(torch.load(ffdnet_path, map_location=device))
        print(f"✓ Loaded model from {ffdnet_path}\n")
    else:
        print(f"Model {ffdnet_path} not found!")
        sys.exit(1)

    if args.image:
        visualize_denoising(model, args.image, args.sigma, device)

    elif args.input_dir:
        batch_denoise(model, args.input_dir, args.output_dir, args.sigma, device)

    else:
        print("Specify --image or --input_dir")


if __name__ == "__main__":
    main()
