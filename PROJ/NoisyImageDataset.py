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

        # Save in (noisy, clear) order so loading assignment matches (noisy_patches, clear_patches)
        torch.save((all_noisy_patches, all_clear_patches), out_path)
        print(f"Saved preprocessed patches to {out_path}, total patches: {all_clear_patches.size(0)}")
        return all_noisy_patches, all_clear_patches

    def __len__(self):
        return len(self.clear_patches) # assuming that clear_image_dir and noisy_image_dir have the same number of images

    def __getitem__(self, idx):
        return self.noisy_patches[idx], self.clear_patches[idx]