# Cat_dog_detection_YOLO_new_metric
# Custom Object Detection with YOLOv5 and Custom Metric

This repository contains the implementation of a custom object detection pipeline using **YOLOv5**. The objective is to detect **cats** and **dogs** from a small, manually labeled image dataset. Additionally, a **custom bounding box similarity metric** has been implemented alongside traditional Intersection over Union (IoU) for improved detection performance.

## Project Overview

The goal of this project is to:
1. Train a YOLOv5 object detection model on a small dataset of cats and dogs.
2. Develop a novel bounding box similarity metric.
3. Evaluate the model and the custom metric, reporting both qualitative and quantitative results.

## Dataset

The dataset used consists of labeled images containing cats and dogs. The images are annotated with bounding boxes, and the data is divided into two classes: **dog** and **cat**.

- **Number of images**: 500 for training, 100 for validation (80%/20% split).
- **Annotations**: Pascal VOC format annotations were converted to YOLO format.

## Setup

### Prerequisites
Make sure to have the following installed:
- Python 3.x
- PyTorch 1.7 or later
- YOLOv5 dependencies (use the requirements file provided below)

### Install Dependencies

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```
### Running the Code
To run the model training and evaluation, use the following command:
```bash
python train.py --img-size 640 --batch-size 16 --epochs 50 --data data.yaml --cfg yolov5s.yaml --weights yolov5s.pt --device 0 --loss-path /path/to/custom/loss.py
```
Custom Metric
This project implements a custom bounding box similarity metric that considers not only the overlap of bounding boxes but also factors like:

Aspect ratio: Difference in width and height ratios between the predicted and ground-truth boxes.

Center alignment: Difference in the central coordinates of the bounding boxes, scaled based on the object size.

The custom metric combines these factors to provide a more comprehensive measure of box similarity than traditional IoU.

Custom Metric Equation
The custom metric is calculated using:
```python
def center_ar_metric_tensor(boxA, boxB, img_size=640):
    # Compute intersection area
    xA = torch.max(boxA[:, 0], boxB[:, 0])
    yA = torch.max(boxA[:, 1], boxB[:, 1])
    xB = torch.min(boxA[:, 2], boxB[:, 2])
    yB = torch.min(boxA[:, 3], boxB[:, 3])

    inter = (xB - xA).clamp(0) * (yB - yA).clamp(0)
    areaA = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
    areaB = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])
    iou = inter / (areaA + areaB - inter + 1e-6)

    # === Compute center x and y differences ===
    cxA = (boxA[:, 0] + boxA[:, 2]) / 2
    cyA = (boxA[:, 1] + boxA[:, 3]) / 2
    cxB = (boxB[:, 0] + boxB[:, 2]) / 2
    cyB = (boxB[:, 1] + boxB[:, 3]) / 2

    dx = torch.abs(cxA - cxB)
    dy = torch.abs(cyA - cyB)

    # === Compute height/width ratios ===
    hA = boxA[:, 3] - boxA[:, 1]
    wA = boxA[:, 2] - boxA[:, 0]
    hB = boxB[:, 3] - boxB[:, 1]
    wB = boxB[:, 2] - boxB[:, 0]

    aspect_scale_x = torch.min(hA / (wA + 1e-6), hB / (wB + 1e-6))
    aspect_scale_y = torch.min(wA / (hA + 1e-6), wB / (hB + 1e-6))

    # === Compute scaled center differences ===
    delta_c = torch.sqrt((dx*aspect_scale_x)**2 +(dy*aspect_scale_y)**2) / np.sqrt(img_size**2 + img_size**2)

    # === Aspect ratio difference ===
    arA = wA / (hA + 1e-6)
    arB = wB / (hB + 1e-6)
    delta_r = torch.abs(arA - arB) / (arA + arB + 1e-6)

    return 0.1* (1 - delta_c) + 0.1*(1 - delta_r)
```
This metric is designed to improve bounding box alignment, especially for objects with varying aspect ratios.
Results
Standard Metrics:
mAP (mean Average Precision): Measures the average precision across all classes.

IoU (Intersection over Union): Measures the overlap between predicted and ground-truth bounding boxes.

Custom Metric:
The custom metric was evaluated alongside the standard metrics to assess its impact on detection performance.

Qualitative Results:
Sample images with bounding boxes will be displayed once the model finishes training. The bounding boxes will show how well the model detects cats and dogs.

Reflection
Performance:
The custom similarity metric is shown to improve the detection of certain objects with specific aspect ratios and center alignment. It is especially useful in cases where the objects are not uniform in size or shape.

Trade-offs:
There is a trade-off between computational complexity and detection accuracy. The additional terms in the custom metric require more computation, but they provide a more robust measure of box alignment and aspect ratio similarity.

Future Improvements:
Class-weighting: Introducing weighted penalties for misclassifications.

Distance-based penalties: Additional penalties for extreme misalignments.
Files
train.py: Main script for training the YOLOv5 model with custom loss functions.

data.yaml: Configuration file for dataset paths.

custom_loss.py: Contains the custom bounding box similarity metric.
