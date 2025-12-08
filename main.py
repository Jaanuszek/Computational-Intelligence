import dataset
import NoisyImageDataset
# from dnCNN import dnCNN_conf

from includes import *

def prepareData():
    dataset.download_dataset()

    dataset.process_images(
        dataset.ORIGINAL_TRAIN_GROUND_TRUTH_DIR,
        dataset.GRAY_TRAIN_DIR,
        sigma=None
    )
    dataset.process_images(
        dataset.ORIGINAL_TEST_GROUND_TRUTH_DIR,
        dataset.GRAY_TEST_DIR,
        sigma=None
    )
    dataset.process_images(
        dataset.ORIGINAL_VALIDATE_GROUND_TRUTH_DIR,
        dataset.GRAY_VALIDATE_DIR,
        sigma=None
    )


    dataset.process_images(
        dataset.ORIGINAL_TRAIN_GROUND_TRUTH_DIR,
        dataset.NOISY_TRAIN_DIR,
        sigma=25,
    )
    dataset.process_images(
        dataset.ORIGINAL_VALIDATE_GROUND_TRUTH_DIR,
        dataset.NOISY_VALIDATE_DIR,
        sigma=25,
    )
    dataset.process_images(
        dataset.ORIGINAL_TEST_GROUND_TRUTH_DIR,
        dataset.NOISY_TEST_DIR,
        sigma=25,
    )

if __name__ == "__main__":
    prepareData()

    