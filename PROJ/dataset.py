from includes import *

MAIN_DIR = "PROJ/"
DATASET_DIR = MAIN_DIR + "datasets/"

ORIGINAL_TEST_GROUND_TRUTH_DIR = DATASET_DIR + "Original/test/ground_truth/"
ORIGINAL_TRAIN_GROUND_TRUTH_DIR = DATASET_DIR + "Original/train/ground_truth/"
ORIGINAL_VALIDATE_GROUND_TRUTH_DIR = DATASET_DIR + "Original/validate/ground_truth/"

GRAY_TEST_DIR= DATASET_DIR + "Gray/test/"
GRAY_TRAIN_DIR= DATASET_DIR + "Gray/train/"
GRAY_VALIDATE_DIR= DATASET_DIR + "Gray/validate/"

NOISY_TEST_DIR= DATASET_DIR + "Noisy/test/"
NOISY_TRAIN_DIR= DATASET_DIR + "Noisy/train/"
NOISY_VALIDATE_DIR= DATASET_DIR + "Noisy/validate/"

PREPROCESSED_TRAIN_DIR = DATASET_DIR + "Preprocessed/train_patches.pt"
PREPROCESSED_TEST_DIR = DATASET_DIR + "Preprocessed/test_patches.pt"
PREPROCESSED_VALIDATE_DIR = DATASET_DIR + "Preprocessed/validate_patches.pt"

def download_dataset():
    if not os.path.isdir(ORIGINAL_TEST_GROUND_TRUTH_DIR):
        path = kagglehub.dataset_download("tarunpathak/natural-images-with-synthetic-noise")
        print("Path to dataset files:", path)
    else:
        print("Dataset already exists locally.")

def add_gaussian_noise(image, sigma):
    """Adds Gaussian noise to an image.
        It assumes that image is normalized to [0, 1].
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

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

