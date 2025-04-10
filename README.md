# Cat_dog_detection_YOLO_new_metric
# Custom Object Detection with YOLOv5

## Overview

This repository contains the implementation of a custom object detection pipeline using **YOLOv5**. The dataset consists of **cats** and **dogs**, and the goal is to create an effective model that detects these objects with a custom bounding box similarity metric.

The custom metric incorporates both **IoU (Intersection over Union)** and additional features such as **center alignment** and **aspect ratio** to improve the detection of bounding boxes.

## Dataset

The dataset used for training and validation is a set of images containing **cats** and **dogs**, with corresponding **bounding box annotations** in the Pascal VOC format.

### Dataset Structure

- **images/**: Folder containing image files (JPEG/PNG).
- **labels/**: Folder containing text files for bounding box annotations (in YOLO format).
- **annotations/**: Folder containing XML files (Pascal VOC format) for ground truth annotations.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- PyTorch 1.7+
- YOLOv5
- Kaggle (for downloading the dataset)

### Install Dependencies

```bash
pip install -r requirements.txt
