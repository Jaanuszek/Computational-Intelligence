import os
from dataset import download_dataset

'''
    The intention of this file is to download and prepare datasets for training and evalutaion.
    This script should be run once.
'''

if __name__ == '__main__':
    download_dataset()