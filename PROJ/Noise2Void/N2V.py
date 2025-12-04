import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2


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

def multi_active_pixels(patch, n_pix, n_rad=5):
    """
    Adding masks to the patch for N2V training
    
    :param patch: Torch tensor of shape (1, H, W)
    :param n_pix: Number of pixels to mask
    :param n_rad: Radius around each pixel to avoid overlapping masks
    """

    # chose random pixels positions 
    idx_aps = np.random.randint(0, patch.shape[1], n_pix)
    idy_aps = np.random.randint(0, patch.shape[2], n_pix)
    id_aps = (idx_aps, idy_aps) # wrap into a tuple
    print(id_aps)

    # random shifts in radius n_rad
    x_shift = np.random.randint(-n_rad // 2 + n_rad % 2, n_rad // 2 + n_rad % 2, n_pix)
    y_shift = np.random.randint(-n_rad // 2 + n_rad % 2, n_rad // 2 + n_rad % 2, n_pix)

    # If is chosed the same pixel, rechoose
    for i in range(len(x_shift)):
        if x_shift[i] == 0 and y_shift[i] == 0:
            shift = np.trim_zeros(np.arange(-n_pix//2 +1, n_pix//2 +1))
            x_shift[i] = np.random.choice(shift[shift != 0], 1)


class N2VDataset(Dataset):
    """
    Docstring for N2VDataset
    images: List of images
    """
    def __init__(self, images, patch_size=64, stride=48):
        self.images = images
        self.patch_size = patch_size
        self.stride = stride
        self.patches = []

        for img in images:
            img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            patches = image_to_patches(img_t, patch_size, stride)
            self.patches.extend(patches)

        self.patches = torch.cat(self.patches, dim=0)

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.patches[idx] # (1, patch_size, patch_size)

if __name__ == "__main__":
    test_img_path = os.path.join(GRAY_DATASET_DIR, 'test', '410936964_75a544fe67_c.jpg')
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    test_img = normalize_img(test_img)
    test_img_t = torch.tensor(test_img, dtype=torch.float32).unsqueeze(0)
    test_img_patches = image_to_patches(test_img_t, patch_size=64, stride=48)

    noisy_test_img = add_gaussian_noise_to_image(test_img, sigma=25)
    noisy_test_img_t = torch.tensor(noisy_test_img, dtype=torch.float32).unsqueeze(0)
    noisy_test_img_patches = image_to_patches(noisy_test_img_t, patch_size=64, stride=48)

    multi_active_pixels(noisy_test_img_patches[0], n_pix=10, n_rad=5)

    # fig, axs = plt.subplots(3, 6, figsize=(15, 7))
    # for i in range(6*3):
    #     axs.ravel()[i].imshow(noisy_test_img_patches[i].squeeze().numpy(), cmap='gray')
    #     axs.ravel()[i].axis('off')
    # fig.tight_layout()
    # plt.show()
    # test_img_arr, train_img_arr, val_img_arr, noisy_test_img_arr, noisy_train_img_arr, noisy_val_img_arr = load_and_process_dataset(GRAY_DATASET_DIR)

    # test_dataset = N2VDataset(noisy_test_img_arr, patch_size=64, stride=48)
    # train_dataset = N2VDataset(noisy_train_img_arr, patch_size=64, stride=48)
    # val_dataset = N2VDataset(noisy_val_img_arr, patch_size=64, stride=48)


