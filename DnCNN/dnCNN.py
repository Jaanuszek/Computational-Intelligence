"""
    dnCNN.py - Defines the DnCNN model architecture and data loaders for training, validation, and testing.
    Also includes training loop with early stopping and a function to denoise images using the trained model.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from includes import *

import dataset
import NoisyImageDataset

class dnCNN_conf:
    clean_train_dir = dataset.GRAY_TRAIN_DIR
    noisy_train_dir = dataset.NOISY_TRAIN_DIR

    clean_val_dir = dataset.GRAY_VALIDATE_DIR
    noisy_val_dir = dataset.NOISY_VALIDATE_DIR

    clean_test_dir = dataset.GRAY_TEST_DIR
    noisy_test_dir = dataset.NOISY_TEST_DIR

    batch_size = 32
    d=21
    receptive_field = 2 * d + 1
    image_hw = 10 * receptive_field
    dimensions = (image_hw, image_hw, 1)

    patch_size = receptive_field
    stride = 3 * receptive_field // 4


transform = transforms.Compose([
    transforms.Resize((dnCNN_conf.image_hw, dnCNN_conf.image_hw)),
    transforms.ToTensor(),
])

train_dataset = NoisyImageDataset.NoisyImageDataset(
    clear_image_dir=dnCNN_conf.clean_train_dir,
    noisy_image_dir=dnCNN_conf.noisy_train_dir,
    sigma=25,
    patch_size=dnCNN_conf.patch_size,
    stride=dnCNN_conf.stride,
    cache_path=dataset.PREPROCESSED_TRAIN_DIR,
    transform=transform
)

val_dataset = NoisyImageDataset.NoisyImageDataset(
    clear_image_dir=dnCNN_conf.clean_val_dir,
    noisy_image_dir=dnCNN_conf.noisy_val_dir,
    sigma=25,
    patch_size=dnCNN_conf.patch_size,
    stride=dnCNN_conf.stride,
    cache_path=dataset.PREPROCESSED_VALIDATE_DIR,
    transform=transform
)

test_dataset = NoisyImageDataset.NoisyImageDataset(
    clear_image_dir=dnCNN_conf.clean_test_dir,
    noisy_image_dir=dnCNN_conf.noisy_test_dir,
    sigma=25,
    patch_size=dnCNN_conf.patch_size,
    stride=dnCNN_conf.stride,
    cache_path=dataset.PREPROCESSED_TEST_DIR,
    transform=transform
    )


train_loader = DataLoader(
    train_dataset,
    batch_size=dnCNN_conf.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
    )
val_loader = DataLoader(
    val_dataset,
    batch_size=dnCNN_conf.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=dnCNN_conf.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

def show_sample_images(dataset, num_examples=3):
    num_examples = min(num_examples, len(dataset))
    indices = torch.randint(0, len(dataset), (num_examples,))

    plt.figure(figsize=(6, 2 * num_examples))
    for i, idx in enumerate(indices):
        noisy_patch, clean_patch = dataset[idx]

        clean_img = clean_patch.squeeze().cpu().numpy()
        noisy_img = noisy_patch.squeeze().cpu().numpy()

        plt.subplot(num_examples, 2, 2 * i + 1)
        plt.imshow(clean_img, cmap='gray')
        plt.title(f"Clean #{i+1}")
        plt.axis('off')

        plt.subplot(num_examples, 2, 2 * i + 2)
        plt.imshow(noisy_img, cmap='gray')
        plt.title(f"Noisy #{i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

class EarlyStopping:
    def __init__(self, tolerance=10, min_delta=0.01):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.min_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.model_dict = None
        self.optim_dict = None
        self.epoch = -1
        
    def __call__(self, val_loss, model_dict, optim_dict, epoch):
        if (self.min_loss - val_loss) > self.min_delta:
            self.counter = 0
            self.min_loss = val_loss
            self.model_dict = model_dict
            self.optim_dict = optim_dict
            self.epoch = epoch
            # save checkpoint of the best model so far
            try:
                torch.save({'model_state_dict': model_dict,
                            'optim_state_dict': optim_dict,
                            'epoch': epoch,
                            'val_loss': val_loss}, 'best_dncnn_checkpoint.pth')
                print(f"Saved best checkpoint (val_loss={val_loss:.6f}) to best_dncnn_checkpoint.pth")
            except Exception as e:
                print("Warning: could not save checkpoint:", e)
        else:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True

class DnCNN(nn.Module):
    def __init__(self, in_channels=1, depth=17, num_filters=64):
        super(DnCNN, self).__init__()
        layers = []
        
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers: Conv + BatchNorm + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer: Conv (no activation or normalization)
        layers.append(nn.Conv2d(in_channels=num_filters, out_channels=in_channels, kernel_size=3, padding=1, bias=False))

        # layers.append(nn.Sigmoid())
        
        self.dncnn = nn.Sequential(*layers)
        # initialize weights for better training stability
        self._initialize_weights()
    
    def forward(self, x):
        # The internal network predicts the noise/residual; return the predicted noise so caller can subtract it
        return self.dncnn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DnCNN(in_channels=1).to(device)
    early_stop = EarlyStopping(tolerance=5, min_delta=0.00001)

    # Hyperparameters
    num_epochs = 5
    learning_rate = 1e-4
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler zmienia lr w czasie treningu
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs) 

    history = {'train_loss': [] ,
            'val_loss': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # x - noisy image, y - clean image
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)  # model predicts the noise/residual
            denoised = x - outputs  # reconstruct clean image
            loss = criterion(denoised, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            history['train_loss'].append(loss.item())
            history['train_loss'].append(loss.item())
        
        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f}", end=" ")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                denoised_val = x_val - outputs
                loss = criterion(denoised_val, y_val)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"| Val Loss: {val_loss:.4f}")

        scheduler.step()
        early_stop(val_loss, model.state_dict(), optimizer.state_dict(), epoch)

        if early_stop.early_stop:
            print("> Stopped training")
    PATH_TO_MODEL = "dnCNN_model.pth"
    # Save the trained model (overwrite any previous)
    torch.save(model.state_dict(), PATH_TO_MODEL)
    model.load_state_dict(torch.load(PATH_TO_MODEL))

    model.load_state_dict(torch.load(PATH_TO_MODEL))
    with torch.no_grad():
        # Fetch a batch of patches
        test_batch = next(iter(test_loader))
        noisy_patches, clean_patches = test_batch
        noisy_patches = noisy_patches.to(device)
        clean_patches = clean_patches.to(device)

        # Predict the noise pattern (model now returns predicted noise)
        predicted_noise_patches = model(noisy_patches).cpu()

        # Optionally compute ground-truth noise for comparison
        gt_noise_patches = (noisy_patches - clean_patches).cpu()

        for idx in range(min(len(noisy_patches), 8)):  # Visualize up to 8 patches
            noisy_patch = noisy_patches[idx].cpu().squeeze().numpy()
            clean_patch = clean_patches[idx].cpu().squeeze().numpy()
            predicted_noise_patch = predicted_noise_patches[idx].squeeze().numpy()
            denoised_patch = noisy_patch - predicted_noise_patch

            # Display the noisy patch, clean image, predicted noise, and denoised patch
            fig, ax = plt.subplots(1, 4, figsize=(12, 4))
            
            ax[0].imshow(noisy_patch, cmap='gray')
            ax[0].set_title('Noisy Patch')
            ax[0].axis('off')

            ax[1].imshow(clean_patch, cmap='gray')
            ax[1].set_title('Clean Patch')
            ax[1].axis('off')

            ax[2].imshow(predicted_noise_patch, cmap='gray')
            ax[2].set_title('Predicted Noise')
            ax[2].axis('off')

            ax[3].imshow(denoised_patch, cmap='gray')
            ax[3].set_title('Denoised Patch')
            ax[3].axis('off')

            plt.show()

    def denoise_image(model, image_path, add_noise=True):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((dnCNN_conf.image_hw, dnCNN_conf.image_hw)),
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0)

        # if add_noise:
        #     noise = torch.from_numpy(gaussian_noise(-20, image_tensor.shape)).float()
        #     noisy_image_tensor = torch.clamp(image_tensor + noise, 0, 1)
        # else:
        noisy_image_tensor = image_tensor.clone()

        device = next(model.parameters()).device
        noisy_image_tensor = noisy_image_tensor.to(device)

        def extract_patches(img_tensor, patch_size, stride):
            C, H, W = img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3]
            patches = []
            positions = []
            # Cover the entire height, including bottom edge
            for i in range(0, H, stride):
                for j in range(0, W, stride):
                    # Compute the end indices
                    end_i = min(i + patch_size, H)
                    end_j = min(j + patch_size, W)
                    
                    patch = img_tensor[:, :, i:end_i, j:end_j]
                    patches.append(patch)
                    positions.append((i, j))
            return patches, positions


        patches, positions = extract_patches(noisy_image_tensor, dnCNN_conf.patch_size, dnCNN_conf.stride)

        denoised_patches = []
        model.eval()
        with torch.no_grad():
            for patch in patches:
                predicted_noise = model(patch)
                denoised_patch = predicted_noise
                denoised_patches.append(torch.clamp(denoised_patch, 0, 1))

        # Reconstruct the image from denoised patches
        C, H, W = noisy_image_tensor.shape[1], noisy_image_tensor.shape[2], noisy_image_tensor.shape[3]
        reconstructed = torch.zeros_like(noisy_image_tensor)
        patch_count = torch.zeros((1, 1, H, W), device=device)

        patch_size = dnCNN_conf.patch_size
        for (i, j), dp in zip(positions, denoised_patches):
            reconstructed[:, :, i:i+patch_size, j:j+patch_size] += dp
            patch_count[:, :, i:i+patch_size, j:j+patch_size] += 1

        safe_patch_count = torch.where(patch_count == 0, torch.ones_like(patch_count), patch_count)
        denoised_image_tensor = reconstructed / safe_patch_count
        denoised_image_tensor = denoised_image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()

        noisy_image_display = noisy_image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(noisy_image_display)
        ax[0].set_title("Noisy Image")
        ax[0].axis('off')

        ax[1].imshow(denoised_image_tensor)
        ax[1].set_title("Denoised Image")
        ax[1].axis('off')
        
        ax[2].imshow(image_tensor.squeeze(0).permute(1, 2, 0))
        ax[2].set_title("Original Image")
        ax[2].axis('off')

        plt.show()
