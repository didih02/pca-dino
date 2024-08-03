
# PCA-Dino

## Overview
PCA-DINO is a project that leverages Principal Component Analysis (PCA) and Dino-ViT.

## Installation

Before you use our code, please install the required libraries if you haven't already. You can install them using the following command:

```
pip3 install scikit-learn torch torchvision numpy
```

## Data Folder (if your dataset doesn't have 'train' and 'val' folders separate):
Ensure that your downloaded dataset follows the specified structure. For instance, when using the Caltech101 dataset, it should be saved in the data/ folder. Please keep the original structure intact when extracting the dataset from the zip folder or any other archive. Do not move or change anything inside, as my program will automatically detect the correct configuration. Example:

```
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

## Split dataset

This script is used to split your dataset into training and validation sets. Below is an example of how to use this script using split_dataset.py:

```
python3 split_dataset.py --data_dir data/caltech101 --output_dir images/caltech101 --train_ratio 0.8
```
Instructions:

**1. Prepare Your Dataset:**
* Ensure your dataset is stored in the correct folder structure.
* In the example above, the dataset is stored in data/caltech101.

**2. Run the Script:**
Use the command shown above to run the script. Replace the paths and ratio as needed:
* --data_dir: The directory where your dataset is stored.
* --output_dir: The directory where you want the split dataset to be saved.
* --train_ratio: The ratio of the dataset to be used for training. In the example above, 0.8 means 80% for training and 20% for validation.

**Notes:**
1. In this example, The dataset is located in data/caltech101.
2. The split datasets will be saved in images/caltech101.
3. Ensure that the directory structure is correctly set up before running the script.
4. You can adjust the --train_ratio parameter to change the proportion of the dataset used for training and validation.
5. By following these steps, you can easily split your dataset into training and validation sets using the split_dataset.py script.

## Extract Features
To extract features using Dino-ViT, you can utilize the dino.py script. Below is an example of how to use this script:
```
python3 dino.py --data_path images/caltech101/ --save_features output/caltech101
```
### Default Configuration
When using this script, the default configuration settings for Dino-ViT are as follows:
* Architecture: vit-small
* patch size: 16
* pre-trained weights: defaults set from previous research
* number workers: 10

### Saving Features
The extracted Dino-ViT features will be saved in the folder specified by the **--save_features argument**. In the example above, the features will be saved in **output/caltech101**.

### Customizing the Configuration
1. Architectures: You can choose from the following architectures: vit_small (default), vit_base, and resnet50. Refer to the [Dino Github Repository](https://github.com/facebookresearch/dino) for more details 
2. Patch Sizes: Available patch sizes are: 8, 16, and 32
3. Pre-trained Weights: Pre-trained weights can be downloaded from this [link](https://github.com/facebookresearch/dino)

By customizing these options, you can adjust the feature extraction process to better suit your specific needs.

## PCA-Dino Classifier
This program allows you to use our classifier on PCA-Dino features. In this research, we use KNN and SVM classifiers. The configuration settings for the classifier can be found in the pca_dino.py code. Below are examples of how to use this script:

```
python3 pca_dino.py --dataset caltech101 --load_features output/ --act_pca True --n_component 20 --svd_solver randomized --float16 True
```
**If you want to use k-fold cross-validation, you can use the pca_dino_.py file with similar arguments. However, if you have limited resources, you can choose a regular validation with an 80% training and 20% testing split (pca_dino.py), as the results from k-fold cross-validation are often comparable.**

### Instructions
1. --dataset caltech101: Set your dataset folder within the images/ directory and name the dataset accordingly, for example, caltech101.
2. --load_features output: Specify the directory where your Dino-ViT extracted features are saved. It will search the folder with the name based on your dataset folder name.
3. --svd_solver: Set your svd_solver on PCA, default is **auto**
4. --float16: Use floating point 16 on your results, basic extract features from Dino-ViT is floating point 32. Default set **False** which means still using Float32
5. Extract Features: Before using this program, ensure that you have extracted features using Dino-ViT and saved them in the correct folder.
6. You can adjust the n_component and svd_solver parameters as needed.
7. Results: The classification results will be saved in the classify_pca_dino folder, which is automatically created when you run the code. If you find a CSV file in the folder, it contains the results of your run. The columns in the CSV file are as follows: the first column is the name of the dataset, the second column is Accuracy, the third column is Top-1 Accuracy, the fourth column is Top-5 Accuracy, the fifth column is the size of the entire dataset after reduction, the sixth column is the number of components, and the last column indicates the timestamp of the run.
   
By following these instructions, you can effectively utilize the PCA-Dino Classifier for your dataset.

## Kernel-PCA and Grid Search
This repository contains code that supports our research on kernel PCA and grid search. You can utilize this code to replicate our experiments or conduct your own. Detailed instructions are provided within the code. **If you want to use k-fold cross-validation on kernel PCA, you can use kernel_pca_.py**

### How to Use
The code for kernel PCA and grid search can be found in this repository.
* Follow the instructions in the code to understand how to configure and run the experiments.
* By using this code, you can explore the effectiveness of kernel PCA and grid search in your research. For any questions or further information, please refer to the comments and documentation provided in the code files.

## Source Dino-ViT:
1. https://github.com/facebookresearch/dino
2. https://github.com/ShirAmir/dino-vit-features
