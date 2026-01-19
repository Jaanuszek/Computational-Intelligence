"""
FFDNet.py - Consist of FFDNet model definition and training utilities.
"""


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from includes import *
import torch.nn.functional as F_torch
import dataset
import NoisyImageDataset

PATH_TO_FFDNET_MODEL = os.path.join(MODEL_DIR, "ffdnet_model.pth")

class FFDNetConfig:
    clean_train_dir = dataset.GRAY_TRAIN_DIR
    noisy_train_dir = dataset.NOISY_TRAIN_DIR
    clean_val_dir = dataset.GRAY_VALIDATE_DIR
    noisy_val_dir = dataset.NOISY_VALIDATE_DIR
    clean_test_dir = dataset.GRAY_TEST_DIR
    noisy_test_dir = dataset.NOISY_TEST_DIR
    
    # Training parameters
    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-3
    
    # Image parameters
    image_size = 256
    patch_size = 64
    stride = 48
    
    # Noise parameters
    sigma_min = 0.0
    sigma_max = 75.0
    sigma_train = 25.0  # Default noise level for training
    
    # Model parameters
    in_channels = 1
    num_features = 64
    num_conv_layers = 15


class PixelShuffle(nn.Module):
    """Space-to-depth operation (inverse of pixel shuffle)"""
    def __init__(self, downscale_factor=2):
        super(PixelShuffle, self).__init__()
        self.downscale_factor = downscale_factor
    
    def forward(self, x):
        # x: [B, C, H, W] -> [B, C*r^2, H/r, W/r]
        batch_size, channels, height, width = x.size()
        r = self.downscale_factor
        
        out_channels = channels * r * r
        out_height = height // r
        out_width = width // r
        
        # Reshape and permute
        x = x.contiguous().view(batch_size, channels, out_height, r, out_width, r)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch_size, out_channels, out_height, out_width)
        
        return x


class FFDNet(nn.Module):
    def __init__(self, in_channels=1, num_features=64, num_conv_layers=15):
        super(FFDNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_features = num_features
        
        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=2)
        
        input_features = in_channels * 4 + 1
        
        layers = []
        
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(input_features, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers: Conv + BN + ReLU
        for _ in range(num_conv_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer: Conv (outputs 4 channels for depth-to-space)
        layers.append(nn.Conv2d(num_features, in_channels * 4, kernel_size=3, padding=1, bias=False))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Depth-to-space transformation (upscale by 2)
        self.depth_to_space = nn.PixelShuffle(upscale_factor=2)
        
        self._initialize_weights()
    
    def forward(self, x, sigma_map):
        """
        Args:
            x: noisy image [B, C, H, W]
            sigma_map: noise level map [B, 1, H, W]
        
        Returns:
            predicted noise [B, C, H, W]
        """
        # Space-to-depth: [B, C, H, W] -> [B, C*4, H/2, W/2]
        x_down = self.space_to_depth(x)
        
        # Downscale noise map to match - use avg_pool2d as in original implementation
        sigma_down = F_torch.avg_pool2d(sigma_map, kernel_size=2)
        
        # Concatenate: [B, C*4+1, H/2, W/2]
        x_cat = torch.cat([x_down, sigma_down], dim=1)
        
        # Process through conv layers
        x_conv = self.conv_layers(x_cat)
        
        # Depth-to-space: [B, C*4, H/2, W/2] -> [B, C, H, W]
        noise = self.depth_to_space(x_conv)
        
        return noise
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class EarlyStopping:
    """Early stopping callback to prevent overfitting"""
    def __init__(self, tolerance=10, min_delta=0.001):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.min_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_dict = None
        self.best_optim_dict = None
        self.best_epoch = -1
    
    def __call__(self, val_loss, model_dict, optim_dict, epoch, save_path=os.path.join(MODEL_DIR, 'best_ffdnet_checkpoint.pth')):
        if (self.min_loss - val_loss) > self.min_delta:
            self.counter = 0
            self.min_loss = val_loss
            self.best_model_dict = model_dict
            self.best_optim_dict = optim_dict
            self.best_epoch = epoch
            
            # Save checkpoint
            try:
                torch.save({
                    'model_state_dict': model_dict,
                    'optim_state_dict': optim_dict,
                    'epoch': epoch,
                    'val_loss': val_loss
                }, save_path)
                print(f"✓ Saved best checkpoint (val_loss={val_loss:.6f}) to {save_path}")
            except Exception as e:
                print(f"Warning: could not save checkpoint: {e}")
        else:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def train_epoch(model, train_loader, criterion, optimizer, device, default_sigma):
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        if len(batch) == 3:
            noisy_patches, clean_patches, sigmas = batch
            # sigmas is a tensor of shape [batch_size]
            # Create noise level map from per-image sigma
            batch_size = noisy_patches.size(0)
            sigma_map = sigmas.view(batch_size, 1, 1, 1).expand(batch_size, 1, noisy_patches.size(2), noisy_patches.size(3)).float().to(device)
            sigma_map = sigma_map / 255.0 # Normalize
        else:
            noisy_patches, clean_patches = batch
            # Create noise level map - normalize sigma to [0,1] range
            batch_size = noisy_patches.size(0)
            sigma_map = torch.full((batch_size, 1, noisy_patches.size(2), noisy_patches.size(3)), 
                                default_sigma / 255.0, dtype=torch.float32, device=device)

        noisy_patches = noisy_patches.to(device)
        clean_patches = clean_patches.to(device)
        
        optimizer.zero_grad()
    
        predicted_noise = model(noisy_patches, sigma_map)
        denoised = noisy_patches - predicted_noise
    
        loss = criterion(denoised, clean_patches)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    return train_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device, sigma, save_samples=False, epoch=0):
    model.eval()
    val_loss = 0.0
    
    # For saving validation samples
    sample_noisy = None
    sample_clean = None
    sample_denoised = None
    
    with torch.no_grad():
        for i, (noisy_patches, clean_patches) in enumerate(val_loader):
            noisy_patches = noisy_patches.to(device)
            clean_patches = clean_patches.to(device)
            
            batch_size = noisy_patches.size(0)
            sigma_map = torch.full((batch_size, 1, noisy_patches.size(2), noisy_patches.size(3)),
                                   sigma / 255.0, dtype=torch.float32, device=device)
            
            predicted_noise = model(noisy_patches, sigma_map)
            denoised = noisy_patches - predicted_noise
            
            loss = criterion(denoised, clean_patches)
            val_loss += loss.item()
            
            # Save first batch samples
            if i == 0 and save_samples:
                sample_noisy = noisy_patches[:4].cpu()
                sample_clean = clean_patches[:4].cpu()
                sample_denoised = denoised[:4].cpu()
    
    # Save validation samples
    if save_samples and sample_noisy is not None and sample_clean is not None and sample_denoised is not None:
        os.makedirs('ffdnet_val_samples', exist_ok=True)
        np.savez(f'ffdnet_val_samples/epoch_{epoch+1:02d}_val_sample.npz',
                 noisy=sample_noisy.numpy(),
                 clean=sample_clean.numpy(),
                 denoised=sample_denoised.numpy())
    
    return val_loss / len(val_loader)

def train_model(model, config, train_loader, val_loader, device) -> dict[str, list]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    early_stop = EarlyStopping(tolerance=8, min_delta=0.0001)
    
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print(f"Noise level (sigma): Dynamic [0, 75]\n")
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config.sigma_train)
        history['train_loss'].append(train_loss)
        
        # Validate
        save_samples = (epoch % 5 == 0) or (epoch == config.num_epochs - 1)
        val_loss = validate_epoch(model, val_loader, criterion, device, config.sigma_train, 
                                  save_samples=save_samples, epoch=epoch)
        history['val_loss'].append(val_loss)
        
        scheduler.step()
        
        print(f"Epoch {epoch + 1}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        early_stop(val_loss, model.state_dict(), optimizer.state_dict(), epoch)
        
        if early_stop.early_stop:
            print(f"\n✓ Early stopping triggered at epoch {epoch + 1}")
            print(f"Best model was at epoch {early_stop.best_epoch + 1} with val_loss={early_stop.min_loss:.6f}")
            break
    
    torch.save(model.state_dict(), PATH_TO_FFDNET_MODEL)
    print(f"\n✓ Final model saved to {PATH_TO_FFDNET_MODEL}")

    return history

def denoise_image(model, noisy_image_tensor, device, sigma):
    model.eval()
    with torch.no_grad():
        noisy_image_tensor = noisy_image_tensor.to(device)
        batch_size = noisy_image_tensor.size(0)
        sigma_map = torch.full((batch_size, 1, noisy_image_tensor.size(2), noisy_image_tensor.size(3)),
                               sigma / 255.0, dtype=torch.float32, device=device)
        predicted_noise = model(noisy_image_tensor, sigma_map)
        denoised_image = noisy_image_tensor - predicted_noise
    return denoised_image.cpu()

def plot_image(path_to_image, model, device, sigma):
    """Plot denoising result for a single image"""
    image = Image.open(path_to_image).convert('L')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    denoised_tensor = denoise_image(model, image_tensor, device, sigma)
    
    noisy_image_np = image_tensor.squeeze().numpy()
    denoised_image_np = denoised_tensor.squeeze().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(noisy_image_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Noisy Image')
    axes[0].axis('off')
    
    axes[1].imshow(denoised_image_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Denoised Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main training loop"""
    config = FFDNetConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = NoisyImageDataset.DynamicNoisyDataset(
        clear_image_dir=config.clean_train_dir,
        sigma_min=0,
        sigma_max=75,
        patch_size=config.patch_size,
        stride=config.stride,
        cache_path=dataset.PREPROCESSED_TRAIN_DIR.replace('train_patches.pt', 'ffdnet_dynamic_train_patches.pt'),
        transform=transform
    )
    
    # Keep validation on fixed sigma=25 for stable comparison
    val_dataset = NoisyImageDataset.NoisyImageDataset(
        clear_image_dir=config.clean_val_dir,
        noisy_image_dir=config.noisy_val_dir,
        sigma=25,
        patch_size=config.patch_size,
        stride=config.stride,
        cache_path=dataset.PREPROCESSED_VALIDATE_DIR.replace('validate_patches.pt', 'ffdnet_validate_patches.pt'),
        transform=transform
    )
    
    test_dataset = NoisyImageDataset.NoisyImageDataset(
        clear_image_dir=config.clean_test_dir,
        noisy_image_dir=config.noisy_test_dir,
        sigma=25,
        patch_size=config.patch_size,
        stride=config.stride,
        cache_path=dataset.PREPROCESSED_TEST_DIR.replace('test_patches.pt', 'ffdnet_test_patches.pt'),
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    model = None
    if not os.path.exists(PATH_TO_FFDNET_MODEL):
        model = FFDNet(
            in_channels=config.in_channels,
            num_features=config.num_features,
            num_conv_layers=config.num_conv_layers
        ).to(device)
        train_model(model, config, train_loader, val_loader, device)
    else:
        model = FFDNet(
            in_channels=config.in_channels,
            num_features=config.num_features,
            num_conv_layers=config.num_conv_layers
        ).to(device)
        model.load_state_dict(torch.load(PATH_TO_FFDNET_MODEL))

    random_image = np.random.choice(os.listdir(dataset.NOISY_VALIDATE_DIR))
    path_to_image = os.path.join(dataset.NOISY_VALIDATE_DIR, random_image)
    plot_image(path_to_image, model, device, sigma=35)
    

if __name__ == "__main__":
    main()

