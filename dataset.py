import shutil
from includes import *

# Get absolute path to the PROJ directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = SCRIPT_DIR
DATASET_DIR = os.path.join(MAIN_DIR, "datasets")

ORIGINAL_DATASET_DIR = os.path.join(DATASET_DIR, "Original/")

ORIGINAL_TEST_DIR = os.path.join(ORIGINAL_DATASET_DIR, "test/")
ORIGINAL_TRAIN_DIR = os.path.join(ORIGINAL_DATASET_DIR, "train/")
ORIGINAL_VALIDATE_DIR = os.path.join(ORIGINAL_DATASET_DIR, "validate/")

ORIGINAL_TEST_GROUND_TRUTH_DIR = os.path.join(ORIGINAL_TEST_DIR, "ground_truth/")
ORIGINAL_TRAIN_GROUND_TRUTH_DIR = os.path.join(ORIGINAL_TRAIN_DIR, "ground_truth/")
ORIGINAL_VALIDATE_GROUND_TRUTH_DIR = os.path.join(ORIGINAL_VALIDATE_DIR, "ground_truth/")

GRAY_TEST_DIR = os.path.join(DATASET_DIR, "Gray/test/")
GRAY_TRAIN_DIR = os.path.join(DATASET_DIR, "Gray/train/")
GRAY_VALIDATE_DIR = os.path.join(DATASET_DIR, "Gray/validate/")

NOISY_TEST_DIR = os.path.join(DATASET_DIR, "Noisy/test/")
NOISY_TRAIN_DIR = os.path.join(DATASET_DIR, "Noisy/train/")
NOISY_VALIDATE_DIR = os.path.join(DATASET_DIR, "Noisy/validate/")

PREPROCESSED_TRAIN_DIR = os.path.join(DATASET_DIR, "Preprocessed/train_patches.pt")
PREPROCESSED_TEST_DIR = os.path.join(DATASET_DIR, "Preprocessed/test_patches.pt")
PREPROCESSED_VALIDATE_DIR = os.path.join(DATASET_DIR, "Preprocessed/validate_patches.pt")

def download_dataset():
    if not os.path.isdir(ORIGINAL_TEST_GROUND_TRUTH_DIR):
        path = kagglehub.dataset_download("tarunpathak/natural-images-with-synthetic-noise")
        print("Path to dataset files:", path)

        copy_dataset_to_project_dir(path)
    else:
        print("Dataset already exists locally.")

def copy_dataset_to_project_dir(path: str):
    if not os.path.exists(DATASET_DIR):
        shutil.copytree(path, DATASET_DIR)
        print(f"Copied dataset to {DATASET_DIR}")
        sort_dataset_files()
        check_for_repeated_dir_names()
        for dir in os.listdir(ORIGINAL_DATASET_DIR):
            change_spaces_to_underscores_in_filenames(os.path.join(ORIGINAL_DATASET_DIR, dir))
    else:
        print(f"Dataset directory {DATASET_DIR} already exists. Skipping copy.")

def check_for_repeated_dir_names():
    if  not os.path.exists(DATASET_DIR):
        return
    
    for dir in os.listdir(ORIGINAL_DATASET_DIR):
        subdir = os.path.join(ORIGINAL_DATASET_DIR, dir)
        if os.path.isdir(subdir) and os.path.isdir(os.path.join(subdir, dir)):
            repeated_dir = os.path.join(subdir, dir)
            for item in os.listdir(repeated_dir):
                shutil.move(os.path.join(repeated_dir, item), subdir)
            os.rmdir(repeated_dir)

def change_spaces_to_underscores_in_filenames(directory):
    for filename in os.listdir(directory):
        if ' ' in filename:
            new_filename = filename.replace(' ', '_')
            os.rename(
                os.path.join(directory, filename),
                os.path.join(directory, new_filename)
            )

def sort_dataset_files():
    """
    test/train/validate goes to Original/<test/train/validate>
    test/train/validate images are changed to gray and go to Gray/<test/train/validate>
    test/train/validate images with noise go to Noisy/<test/train/validate>
    """

    original_dir_map : dict = {
        'test': ORIGINAL_TEST_DIR,
        'train': ORIGINAL_TRAIN_DIR,
        'validate': ORIGINAL_VALIDATE_DIR,
    }

    Noisy_dir_map : dict = {
        'test': NOISY_TEST_DIR,
        'train': NOISY_TRAIN_DIR,
        'validate': NOISY_VALIDATE_DIR,
    }

    # img folders to Original/
    for dir in os.listdir(DATASET_DIR):
        if dir.lower() in original_dir_map.keys():
            shutil.move(
                os.path.join(DATASET_DIR, dir), original_dir_map[dir.lower()])

def add_gaussian_noise(image, sigma_255):
    sigma = sigma_255 / 255.0
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

def normalize_img(img):
    return img.astype(np.float32) / 255.0

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=1.0)

def load_image(path):
    img = Image.open(path).convert('L')
    return np.array(img).astype(np.float32) / 255.0

def process_images(input_dir, output_dir, sigma=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(os.listdir(output_dir)) > 0:
        print(f"Noisy images already exist in {output_dir}. Skipping processing.")
        return

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename)

            image = cv2.imread(in_path)
            if image is None:
                print(f"Warning: Could not read image {in_path}. Skipping.")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            gray_norm = gray.astype(np.float32) / 255.0  # Normalize to [0, 1]

            if sigma is not None:
                gray_norm_noisy = add_gaussian_noise(gray_norm, sigma)
            else:
                gray_norm_noisy = gray_norm

            gray_norm_noisy = (gray_norm_noisy * 255).astype(np.uint8)  # Convert back to [0, 255]
            cv2.imwrite(out_path, gray_norm_noisy)

def prepareData():
    download_dataset()

    process_images(
        ORIGINAL_TEST_GROUND_TRUTH_DIR,
        GRAY_TEST_DIR,
        sigma=None,
    )
    process_images(
        ORIGINAL_TRAIN_GROUND_TRUTH_DIR,
        GRAY_TRAIN_DIR,
        sigma=None,
    )
    process_images(
        ORIGINAL_VALIDATE_GROUND_TRUTH_DIR,
        GRAY_VALIDATE_DIR,
        sigma=None,
    )
    process_images(
        ORIGINAL_TEST_GROUND_TRUTH_DIR,
        NOISY_TEST_DIR,
        sigma=25,
    )
    process_images(
        ORIGINAL_TRAIN_GROUND_TRUTH_DIR,
        NOISY_TRAIN_DIR,
        sigma=25,
    )
    process_images(
        ORIGINAL_VALIDATE_GROUND_TRUTH_DIR,
        NOISY_VALIDATE_DIR,
        sigma=25,
    )