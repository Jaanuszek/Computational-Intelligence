import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
from simple_unet import SimpleUNet


# I will not resue the code from the previous models dnCNN and FFDNet,
# It's just to messy, I will do that once again here

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, '..'))

GRAY_DATASET_DIR = os.path.join(ROOT_DIR, 'datasets', 'Gray')


def normalize_img(img):
    return img.astype(np.float32) / 255.0

def add_gaussian_noise(img_arr, sigma=25):
    """
    Docstring for add_gaussian_noise
    
    :param img_arr: Normalized image array (0-1 scale)
    :param sigma: Normalized standard deviation of the Gaussian noise to be added (0-255 scale)
    """
    noisy_arr = []
    for img in img_arr:
        noise = np.random.normal(0, sigma/255.0, img.shape).astype(np.float32)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0.0, 1.0)
        noisy_arr.append(noisy_img)
    return noisy_arr

def add_gaussian_noise_to_image(img, sigma=25):
    """
    Docstring for add_gaussian_noise_to_image
    
    :param img: Normalized image (0-1 scale)
    :param sigma: Normalized standard deviation of the Gaussian noise to be added (0-255 scale)
    """
    noise = np.random.normal(0, sigma/255.0, img.shape).astype(np.float32)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    return noisy_img

def load_and_process_dataset(path):
    """
        IMAGES ARE NORMALIZED
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path {path} does not exist.")
    
    test_img_arr = []
    noisy_test_img_arr = []
    train_img_arr = []
    noisy_train_img_arr = []
    val_img_arr = []
    noisy_val_img_arr = []

    dir_list = os.listdir(path)
    for dir in dir_list:
        full_dir = os.path.join(path, dir)
        if not os.path.isdir(full_dir):
            raise ValueError(f"Expected directory but found file: {full_dir}")
        for file in os.listdir(full_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(full_dir, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = normalize_img(img)
                noisy_img = add_gaussian_noise_to_image(img, sigma=25)
                if img is None:
                    print(f"Warning: Unable to read image {img_path}. Skipping.")
                    continue
                if dir == 'test':
                    test_img_arr.append(img)
                    noisy_test_img_arr.append(noisy_img)
                elif dir == 'train':
                    train_img_arr.append(img)
                    noisy_train_img_arr.append(noisy_img)
                elif dir == 'validate':
                    val_img_arr.append(img)
                    noisy_val_img_arr.append(noisy_img)

    return test_img_arr, train_img_arr, val_img_arr, noisy_test_img_arr, noisy_train_img_arr, noisy_val_img_arr        

def image_to_patches(img, patch_size, stride):
    """ Convert a single image to patches """

    assert img.dim() == 3, "Input image tensor must have 3 dimensions (C, H, W)"

    c, h, w = img.shape
    patches = img.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.contiguous().view(-1, c, patch_size, patch_size)
    return patches

def multi_active_pixels(patch, n_pix, n_rad=5, use_gradient=False, gradient_bias=0.7):
    """
    Adding masks to the patch for N2V training
    
    :param patch: Torch tensor of shape (H, W) or (1, H, W)
    :param n_pix: Number of pixels to mask
    :param n_rad: Radius around each pixel to avoid overlapping masks
    :param use_gradient: If True, use gradient-based masking
    :param gradient_bias: Fraction of pixels to sample from high-gradient regions (0-1)
    """
    # Ensure patch has 3 dimensions (1, H, W)
    if patch.ndim == 2:
        patch = patch.unsqueeze(0)
    
    h, w = patch.shape[1], patch.shape[2]
    
    if use_gradient:
        patch_np = patch[0].cpu().numpy()
        
        grad_x = cv2.Sobel(patch_np, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch_np, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_flat = grad_magnitude.flatten()
        
        epsilon = 1e-8
        grad_prob = (grad_flat + epsilon) / (grad_flat.sum() + epsilon)
        
        n_gradient = int(n_pix * gradient_bias)
        n_random = n_pix - n_gradient
        
        sampled_indices = []
        
        if n_gradient > 0:
            gradient_samples = np.random.choice(
                len(grad_flat),
                size=n_gradient,
                replace=False,
                p=grad_prob
            )
            sampled_indices.extend(gradient_samples)
        
        if n_random > 0:
            all_indices = set(range(len(grad_flat)))
            available = list(all_indices - set(sampled_indices))
            if len(available) > 0:
                random_samples = np.random.choice(available, size=min(n_random, len(available)), replace=False)
                sampled_indices.extend(random_samples)
        
        # Convert flat indices to 2D coordinates
        sampled_indices = np.array(sampled_indices)
        idx_aps = sampled_indices // w
        idy_aps = sampled_indices % w
    else:
        # RANDOM MASKING (original)
        idx_aps = np.random.randint(0, h, n_pix)
        idy_aps = np.random.randint(0, w, n_pix)
    
    # Active pixels indices tuple
    id_aps = (0, idx_aps, idy_aps)

    # random shifts in radius n_rad
    x_shift = np.random.randint(-n_rad // 2 + n_rad % 2, n_rad // 2 + n_rad % 2, n_pix)
    y_shift = np.random.randint(-n_rad // 2 + n_rad % 2, n_rad // 2 + n_rad % 2, n_pix)

    # If is chosed the same pixel, rechoose
    for i in range(len(x_shift)):
        if x_shift[i] == 0 and y_shift[i] == 0:
            shift = np.trim_zeros(np.arange(-n_pix//2 +1, n_pix//2 +1))
            if len(shift[shift != 0]) > 0:
                x_shift[i] = int(np.random.choice(shift[shift != 0], 1)[0])

    # neighbor pixels indices to replace current active pixels
    n_idx = idx_aps + x_shift 
    n_idy = idy_aps + y_shift
    # wrap around
    n_idx = n_idx % h
    n_idy = n_idy % w
    # tuple of final x,y indices of neighbour pixels
    n_id = (0, n_idx, n_idy) 

    cp_patch = patch.clone()
    cp_patch[id_aps] = patch[n_id]

    mask = torch.ones_like(patch)
    mask[id_aps] = 0.0

    return cp_patch, mask

class N2VDataset(Dataset):
    """
    Docstring for N2VDataset
    images: List of images
    """
    def __init__(self, images, patch_size=64, stride=48, perc_active=2, n_rad=5, use_gradient=False, gradient_bias=0.7):
        self.images = images
        self.patch_size = patch_size
        self.stride = stride
        self.perc_active = perc_active
        self.n_rad = n_rad
        self.use_gradient = use_gradient
        self.gradient_bias = gradient_bias
        self.patches = []

        for img in images:
            img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            patches = image_to_patches(img_t, patch_size, stride)
            self.patches.extend(patches)

        self.patches = torch.stack(self.patches, dim=0)
        
        # Calculate number of active pixels
        total_num_pixels = patch_size * patch_size
        self.n_activepixels = int(np.floor((total_num_pixels * perc_active) / 100))
        
        masking_mode = "GRADIENT-BASED" if use_gradient else "RANDOM"
        print(f"N2VDataset: {len(self.patches)} patches, masking={masking_mode}")

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        original_patch = self.patches[idx].squeeze(0)  # Remove channel dim: (H, W)
        # Generate corrupted patch and mask on-the-fly
        corrupted_patch, mask = multi_active_pixels(
            original_patch, 
            n_pix=self.n_activepixels, 
            n_rad=self.n_rad,
            use_gradient=self.use_gradient,
            gradient_bias=self.gradient_bias
        )
        # Squeeze to (H, W) if needed
        corrupted_patch = corrupted_patch.squeeze(0) if corrupted_patch.ndim == 3 else corrupted_patch
        mask = mask.squeeze(0) if mask.ndim == 3 else mask
        
        # Add channel dimension back: (1, H, W)
        corrupted_patch = corrupted_patch.unsqueeze(0)
        original_patch = original_patch.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        return corrupted_patch, original_patch, mask

def n2v_train(model, criterion, optimizer, data_loader, device):
    model.train()
    accuracy = 0
    loss = 0

    for dl in tqdm(data_loader):
        X, y, mask = dl[0].to(device), dl[1].to(device), dl[2].to(device)
        optimizer.zero_grad()
        output = model(X)
        
        blind_spot_mask = (1 - mask)
        loss_value = criterion(output * blind_spot_mask, y * blind_spot_mask)
        loss_value.backward()
        optimizer.step()

        with torch.no_grad():
            ypred = (output.detach().cpu().numpy()).astype(float)

        loss += loss_value.item()
        accuracy += np.sqrt(np.mean((y.cpu().numpy().ravel( ) - ypred.ravel() )**2))
    loss /= len(data_loader)
    accuracy /= len(data_loader)

    return loss, accuracy

def n2v_evaluate(model, criterion, data_loader, device):
    model.eval()
    accuracy = 0
    loss = 0

    for dl in tqdm(data_loader):
        X, y, mask = dl[0].to(device), dl[1].to(device), dl[2].to(device)

        with torch.no_grad():
            output = model(X)
            
            blind_spot_mask = (1 - mask)
            loss_value = criterion(output * blind_spot_mask, y * blind_spot_mask)

            ypred = (output.detach().cpu().numpy()).astype(float)

        loss += loss_value.item()
        accuracy += np.sqrt(np.mean((y.cpu().numpy().ravel( ) - ypred.ravel() )**2))

    loss /= len(data_loader)
    accuracy /= len(data_loader)
    return loss, accuracy

if __name__ == "__main__":
    test_img_arr, train_img_arr, val_img_arr, noisy_test_img_arr, noisy_train_img_arr, noisy_val_img_arr = load_and_process_dataset(GRAY_DATASET_DIR)

    test_dataset = N2VDataset(noisy_test_img_arr, patch_size=64, stride=48, perc_active=2, n_rad=5, use_gradient=False, gradient_bias=0.7)
    train_dataset = N2VDataset(noisy_train_img_arr, patch_size=64, stride=48, perc_active=2, n_rad=5, use_gradient=False, gradient_bias=0.7)
    val_dataset = N2VDataset(noisy_val_img_arr, patch_size=64, stride=48, perc_active=2, n_rad=5, use_gradient=False, gradient_bias=0.7)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    network = SimpleUNet(in_channels=1, out_channels=1, base_channels=64).to(device)

    lr = 0.0001
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    n_epochs = 5
    batch_size = 64

    # g = torch.Generator()
    # g.manual_seed(0)

    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    train_loss_history = np.zeros(n_epochs)
    train_accuracy_history = np.zeros(n_epochs)
    test_loss_history = np.zeros(n_epochs)
    test_accuracy_history = np.zeros(n_epochs)
    
    best_loss = float('inf')
    checkpoint_dir = os.path.join(CURR_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(n_epochs):
        train_loss, train_accuracy = n2v_train(network, criterion, optimizer, train_loader, device)
        train_loss_history[epoch] = train_loss
        train_accuracy_history[epoch] = train_accuracy
        test_loss, test_accuracy = n2v_evaluate(network, criterion, test_loader, device)
        test_loss_history[epoch] = test_loss
        test_accuracy_history[epoch] = test_accuracy

        print(f'''Epoch {epoch+1}/{n_epochs} --, 
        Training Loss {train_loss:.4f},     Training Accuracy {train_accuracy:.4f}, 
        Test Loss {test_loss:.4f},     Test Accuracy {test_accuracy:.4f} ''')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'n2v_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_path = os.path.join(CURR_DIR, 'best_n2v_model.pth')
            torch.save(network.state_dict(), best_model_path)
            print(f'Best model saved with test loss: {best_loss:.4f}')

    # save final model
    model_path = os.path.join(CURR_DIR, 'n2v_unet_model.pth')
    torch.save(network.state_dict(), model_path)
    print(f'Final model saved: {model_path}')

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(train_loss_history)
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(train_accuracy_history)
    axs[0, 1].set_title('Train Accuracy (RMSE)')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(test_loss_history)
    axs[1, 0].set_title('Test Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(test_accuracy_history)
    axs[1, 1].set_title('Test Accuracy (RMSE)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('RMSE')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()