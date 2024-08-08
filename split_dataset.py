#this program for divide data train and val
import argparse
import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import random_split

import argparse

# Function to save images to a specified directory for example images/ folder
def save_images(dataset, dataset_type, root_dir, type):
    for i, (img, label) in enumerate(dataset):
        # Get the class label name
        class_name = dataset.dataset.classes[label]
        # Define the path to save the image
        save_dir = os.path.join(root_dir, dataset_type, class_name)
        os.makedirs(save_dir, exist_ok=True)
        # Save the image
        img_path = os.path.join(save_dir, f'{i}.{type}')
        transforms.ToPILImage()(img).save(img_path)

def split(data_dir, train_ratio, output_dir, type):
    # Define the transform to preprocess the data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load the dataset which save on data/ folder
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Define the split ratio for train and val
    train_ratio = 0.8
    # val_ratio = 0.2

    # Calculate the number of samples for train and val
    num_train = int(len(dataset) * train_ratio)
    num_val = len(dataset) - num_train

    # Split the dataset into train and val sets
    train_set, val_set = random_split(dataset, [num_train, num_val])

    # Save train and val images to folders
    save_images(train_set, 'train', output_dir, type)
    save_images(val_set, 'val', output_dir, type)

    print(f"Images have been saved to the directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Split Dataset')
    parser.add_argument('--data_dir', default=None, type=str, help='Set your dataset folder')
    parser.add_argument('--output_dir', default="images", type=str, help='Set your dataset output after splitting, if not existing, it will make a folder based on your outpur_dir argument')
    parser.add_argument('--train_ratio', default=0.8, type=float, help='Set your train ratio')
    parser.add_argument('--type_images', default="png", type=str, help='Set your type image which want you save, ex: jpg, png, jpeg')
    args = parser.parse_args()

split(args.data_dir, args.train_ratio, args.output_dir, args.type_images)

#make sure your download data with this structure, for example Caltech101 where dataset save on folder data/
# data/
#     caltech101/
#             accordion/
#                 image_0001.jpg
#                 ...
#             airplane/
#                 image_0001.jpg
#                 ...
#             ...

#how to use this code, for example
#python3 split_dataset.py --data_dir data/caltech101 --output_dir images/caltech101 --train_ratio 0.8 --type_images png