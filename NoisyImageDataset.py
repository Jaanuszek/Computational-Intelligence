from includes import *
from dataset import add_gaussian_noise


class NoisyImageDataset(Dataset):
    """ Consumes only gray scale images, and already noisy images """

    def __init__(self, clear_image_dir, noisy_image_dir, sigma, patch_size, stride, cache_path, transform=None):
        self.clear_image_dir = clear_image_dir
        self.noisy_image_dir = noisy_image_dir
        self.sigma = sigma
        self.patch_size = patch_size
        self.stride = stride
        self.cache_path = cache_path
        self.transform = transform

        if len(os.listdir(clear_image_dir)) != len(os.listdir(noisy_image_dir)):
            raise ValueError("The number of clear images must match the number of noisy images.")
        
        if os.path.exists(cache_path):
            print("Loading preprocessed patches from cache:", cache_path)
            self.noisy_patches, self.clear_patches = torch.load(cache_path)
        else:
            print("Cache not found. Processing images...")
            self.noisy_patches, self.clear_patches = self.preprocess_and_save(cache_path)
        
    def preprocess_and_save(self, out_path):
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

        clear_paths = sorted([os.path.join(self.clear_image_dir, f) for f in os.listdir(self.clear_image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        noisy_paths = sorted([os.path.join(self.noisy_image_dir, f) for f in os.listdir(self.noisy_image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        all_clear_patches = []
        all_noisy_patches = []

        for cpath, npath in zip(clear_paths, noisy_paths):
            clear_image = self.transform(Image.open(cpath))
            noisy_image = self.transform(Image.open(npath))

            clear_patches = clear_image.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
            noisy_patches = noisy_image.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)

            clear_patches = clear_patches.contiguous().view(-1, 1, self.patch_size, self.patch_size)
            noisy_patches = noisy_patches.contiguous().view(-1, 1, self.patch_size, self.patch_size)

            all_clear_patches.append(clear_patches)
            all_noisy_patches.append(noisy_patches)

        all_clear_patches = torch.cat(all_clear_patches)
        all_noisy_patches = torch.cat(all_noisy_patches)

        torch.save((all_noisy_patches, all_clear_patches), out_path)
        print(f"Saved preprocessed patches to {out_path}, total patches: {all_clear_patches.size(0)}")
        return all_noisy_patches, all_clear_patches

    def __len__(self):
        return len(self.clear_patches)

    def __getitem__(self, idx):
        return self.noisy_patches[idx], self.clear_patches[idx]


class DynamicNoisyDataset(Dataset):
    """
    Dataset that loads clean images/patches and adds Gaussian noise dynamically on-the-fly.
    This allows training the model on a range of noise levels (e.g., [0, 75]).
    """
    def __init__(self, clear_image_dir, sigma_min=0, sigma_max=75, patch_size=64, stride=48, cache_path=None, transform=None):
        self.clear_image_dir = clear_image_dir
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        if cache_path and os.path.exists(cache_path):
            print("Loading clean patches from cache:", cache_path)
            self.clear_patches = torch.load(cache_path)
        else:
            print("Processing clean images...")
            self.clear_patches = self.preprocess_and_save(cache_path)

    def preprocess_and_save(self, out_path):
        if out_path and not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

        clear_paths = sorted([os.path.join(self.clear_image_dir, f) for f in os.listdir(self.clear_image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

        all_clear_patches = []

        for cpath in clear_paths:
            img = Image.open(cpath).convert('L')
            clear_image = self.transform(img)
            
            patches = clear_image.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
            patches = patches.contiguous().view(-1, 1, self.patch_size, self.patch_size)
            
            all_clear_patches.append(patches)

        all_clear_patches = torch.cat(all_clear_patches)

        if out_path:
            torch.save(all_clear_patches, out_path)
            print(f"Saved clean patches to {out_path}, total patches: {all_clear_patches.size(0)}")
        
        return all_clear_patches

    def __len__(self):
        return len(self.clear_patches)

    def __getitem__(self, idx):
        clean_patch = self.clear_patches[idx]
        
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        
        noise = torch.randn_like(clean_patch) * (sigma / 255.0)
        noisy_patch = clean_patch + noise
        
        return noisy_patch, clean_patch, sigma