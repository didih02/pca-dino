
# PCA-DIno

## Overview
PCA-DINO is a project that leverages Principal Component Analysis (PCA) and Dino-ViT.

## Installation

Before you use our code, please install the required libraries if you haven't already. You can install them using the following command:

```bash
pip3 install scikit-learn torch torchvision numpy
```

## Source:
1. https://github.com/facebookresearch/dino
2. https://github.com/ShirAmir/dino-vit-features

## Data Folder (if your dataset doesn't have 'train' and 'val' folders separate):
Ensure that your downloaded dataset follows the specified structure. For instance, when using the Caltech101 dataset, it should be saved in the data/ folder. Please keep the original structure intact when extracting the dataset from the zip folder or any other archive. Do not move or change anything inside, as my program will automatically detect the correct configuration.

```bash
data/
    caltech101/
        101_ObjectCategories/
            accordion/
                image_0001.jpg
                ...
            airplane/
                image_0001.jpg
                ...
            ...
```
