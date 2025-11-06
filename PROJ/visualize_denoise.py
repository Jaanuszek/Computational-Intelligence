import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os


class DnCNN(nn.Module):
    """Minimal DnCNN to match training architecture (grayscale)
    This matches the model defined in `dnCNN.py` used for training.
    """
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
        # returns predicted noise
        return self.dncnn(x)


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


def visualize(before, after, out_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(before.squeeze(), cmap='gray')
    ax[0].set_title('Noisy')
    ax[0].axis('off')

    ax[1].imshow(after.squeeze(), cmap='gray')
    ax[1].set_title('Denoised')
    ax[1].axis('off')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Saved visualization to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize denoising result using a saved DnCNN model')
    parser.add_argument('image', help='Path to noisy image (grayscale or RGB)')
    parser.add_argument('model', help='Path to saved model state_dict (pth)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to run the model on')
    parser.add_argument('--resize', type=int, default=None, help='Optional: resize the image to square size before denoising')
    parser.add_argument('--out', default=None, help='Optional: path to save the visualization PNG')
    args = parser.parse_args()

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    print(os.getcwd())
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    img = load_image_gray(args.image, resize=args.resize)
    x = prepare_tensor(img, device)

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

    visualize(before_np, after_np, out_path=args.out)


if __name__ == '__main__':
    main()
