# Synthetic Dataset Generator for Object Segmentation
This script generates synthetic datasets for object segmentation tasks by combining extracted objects from input images with randomly generated backgrounds. It produces augmented images along with COCO-compliant annotations.

![test2](https://github.com/user-attachments/assets/ee086a80-5033-4b74-b1cd-8574aae663af)
![synthetic_15](https://github.com/user-attachments/assets/d154c1fc-92e6-4859-ba5d-acb8f1b7ac79)

# Features
Random Background Generation: Creates unique, multi-colored backgrounds.
Object Extraction: Isolates objects from images using HSV color thresholds and morphological operations.
Data Augmentation: Applies transformations like scaling, rotation, color adjustments, Gaussian blur, and noise to extracted objects.
Synthetic Image Composition: Merges augmented objects with backgrounds.
COCO Annotation Output: Generates COCO-format annotations, including bounding boxes, segmentation masks, and metadata.

# How It Works
Input: Reads images from the specified input folder.
Object Isolation: Extracts objects and their masks from each image.
Augmentation: Applies various transformations to extracted objects.
Composition: Places the transformed objects onto random backgrounds.
Output: Saves the synthetic images and their corresponding COCO annotations.

# Usage
Organize your input images in a folder (default: ./images).
Specify the output folder for synthetic images (default: ./o_images).
Define the output path for COCO annotations (default: ./annotations.json).

# Parameters
input_folder: Path to the folder containing the source images.
output_folder: Path to save the generated synthetic images.
coco_file: Path to save the COCO annotations.
count: Number of synthetic images to generate per input image.

# Requirements
Python 3.x
OpenCV
NumPy
pycocotools

# Example
Given an input folder with images, this script generates synthetic images with diverse backgrounds and transformations, suitable for training object detection or segmentation models.


