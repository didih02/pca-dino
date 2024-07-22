
# PCA-DIno

## Overview
PCA-DINO is a project that leverages Principal Component Analysis (PCA) and Dino-ViT.

## Installation

Before you use our code, please install the required libraries if you haven't already. You can install them using the following command:

```bash
pip3 install scikit-learn torch torchvision numpy
```

## Data Folder (if your dataset doesn't have 'train' and 'val' folders separate):
Ensure that your downloaded dataset follows the specified structure. For instance, when using the Caltech101 dataset, it should be saved in the data/ folder. Please keep the original structure intact when extracting the dataset from the zip folder or any other archive. Do not move or change anything inside, as my program will automatically detect the correct configuration. Example:

```bash
data/
    caltech101/
            accordion/
                image_0001.jpg
                ...
            airplane/
                image_0001.jpg
                ...
            ...
```

## split_dataset.py

This script is used to split your dataset into training and validation sets. Below is an example of how to use this script:

```bash
python3 split_dataset.py --data_dir data/caltech101 --output_dir images/caltech101 --train_ratio 0.8
```
# Instructions
Prepare Your Dataset:

Ensure your dataset is stored in the correct folder structure. In the example above, the dataset is stored in data/caltech101.
Run the Script:

Use the command shown above to run the script. Replace the paths and ratio as needed:
--data_dir: The directory where your dataset is stored.
--output_dir: The directory where you want the split dataset to be saved.
--train_ratio: The ratio of the dataset to be used for training. In the example above, 0.8 means 80% for training and 20% for validation.

## Source Dino-ViT:
1. https://github.com/facebookresearch/dino
2. https://github.com/ShirAmir/dino-vit-features
