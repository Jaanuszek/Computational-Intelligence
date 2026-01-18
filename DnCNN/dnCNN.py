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

PATH_TO_DNCNN_MODEL = os.path.join(MODEL_DIR, "dnCNN_model.pth")

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
    num_workers=0,
    pin_memory=True
    )
val_loader = DataLoader(
    val_dataset,
    batch_size=dnCNN_conf.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=dnCNN_conf.batch_size,
    shuffle=False,
    num_workers=0,
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
            try:
                torch.save({'model_state_dict': model_dict,
                            'optim_state_dict': optim_dict,
                            'epoch': epoch,
                            'val_loss': val_loss}, os.path.join(MODEL_DIR, 'best_dncnn_checkpoint.pth'))
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

def train_model(model, num_epochs, learning_rate, device, early_stop) -> dict[str, list]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

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
        history['val_loss'].append(val_loss)

        scheduler.step()
        early_stop(val_loss, model.state_dict(), optimizer.state_dict(), epoch)

        if early_stop.early_stop:
            print("> Stopped training")
            break

    torch.save(model.state_dict(), PATH_TO_DNCNN_MODEL)
    return history

def denoise_image(model, noisy_image_tensor, device):
    model.eval()
    with torch.no_grad():
        noisy_image_tensor = noisy_image_tensor.to(device)
        predicted_noise = model(noisy_image_tensor)
        denoised_image = noisy_image_tensor - predicted_noise
    return denoised_image.cpu()

def plot_image(path_to_image, model, device):

    noisy_image = Image.open(path_to_image).convert('L')
    denoised = denoise_image(model, transform(noisy_image).unsqueeze(0), device)

    fix, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(noisy_image, cmap='gray')
    axes[0].set_title('Noisy Image')
    axes[0].axis('off')

    axes[1].imshow(denoised.squeeze().cpu().numpy(), cmap='gray')
    axes[1].set_title('DnCNN Denoised Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(PATH_TO_DNCNN_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(PATH_TO_DNCNN_MODEL):
        model = DnCNN(in_channels=1).to(device)
        early_stop = EarlyStopping(tolerance=5, min_delta=0.00001)
        history = train_model(model, num_epochs=5, learning_rate=1e-3, device=device, early_stop=early_stop)
    else:
        model = DnCNN(in_channels=1).to(device)
        model.load_state_dict(torch.load(PATH_TO_DNCNN_MODEL))
        print(f"✓ Loaded model from {PATH_TO_DNCNN_MODEL}\n")

    random_image = np.random.choice(os.listdir(dataset.NOISY_VALIDATE_DIR))
    path_to_image = os.path.join(dataset.NOISY_VALIDATE_DIR, random_image)
    plot_image(path_to_image, model, device)
