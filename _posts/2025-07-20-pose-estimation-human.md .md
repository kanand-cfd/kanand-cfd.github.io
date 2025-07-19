---
layout: post
title: "Human Pose Estimation from Images"
math: true
date: 2025-07-19 08:00:00 +0200
categories: [pytorch]
tags: [ml, computer-vision, convolutional-neural-networks]
---

# Human Pose Estimation from Images

*By Karan Anand, PhD*

---

## 🧐 Motivation

Human pose estimation is a fundamental task in computer vision, aiming to predict key joint positions (like head, shoulders, wrists, etc.) from an image. This project explores a bottom-up keypoint detection approach using heatmaps and builds the training pipeline from scratch using COCO keypoints data.

---

## 📊 Project Objectives

* Predict 17 body keypoints per person using the COCO format
* Generate ground truth heatmaps as Gaussian blobs
* Train a CNN-based pose estimation model using dual loss:

  * Heatmap BCE loss
  * Coordinate regression loss (via soft-argmax)
* Evaluate using the PCK\@0.2 metric

---

## 📂 Dataset Preparation

* Subset extracted from COCO Keypoints 2017
* Images filtered to include at least one visible person with valid keypoints
* Used subsets of 200 and 2000 images for experiments
* Input image size: **256x192**
* Heatmap resolutions: **96x72** and **192x256** (depending on decoder)

---

## 🔧 Architecture

### 1. **Encoder**

* Pretrained ResNet18 used to extract deep visual features

### 2. **Decoders**

* **SimplePoseNet**: 3-layer conv-transpose decoder
* **UNetPoseNet**: U-Net style decoder with skip connections

### 3. **Heatmap Decoder Output**

* Predicts 17-channel heatmaps
* Soft-argmax used to decode continuous keypoint coordinates

---

## ⚖️ Loss Functions

* **Heatmap Loss**: `BCEWithLogitsLoss`
* **Coordinate Loss**: `L1Loss` (between soft-argmax output and ground truth)
* **Dual Loss**: `Total = heatmap_loss + 0.1 * coord_loss`
* **Joint-wise weighting** added to emphasize harder joints (wrists, ankles)

---

## 📊 Evaluation Metrics

* **PCK\@0.2**: Percentage of Correct Keypoints within 20% of torso size
* **Visualizations**: Ground truth vs predicted keypoints overlayed on images

---

## 📊 Results

* Best performance: **PCK\@0.2 ≈ 0.51** (on 2000 images, UNet decoder)
* Training for 30 epochs yielded visible improvement in heatmap localization
* Soft-argmax significantly improved convergence and keypoint sharpness

---

## 🤔 Key Observations

* Ground truth heatmaps must match model output resolution to avoid misalignment
* SimplePoseNet was faster but underfit difficult joints
* UNet decoder provided sharper localization but trained slower
* Coordinate loss helps guide ambiguous blobs to correct position

---

## ⚠️ Known Issues

* **Multi-person ambiguity**: Only the first visible person is supervised; keypoints may sometimes be projected onto other individuals in crowded scenes
* **Pose structure mismatch**: Even when keypoints are in the right area, the relative arrangement often does not match the skeleton
* **Heatmap upsampling artifacts**: Early experiments with mismatched GT and output resolutions caused training instability

---

## 🎓 What I Learned

* Combining heatmap supervision with coordinate decoding improves spatial precision
* U-Net style decoders can enhance weak joints but require careful tuning
* Soft-argmax is a differentiable, effective way to extract coordinates
* Visual debugging and per-joint metric tracking are essential

---

## ✏️ Next Steps

* Add structural priors or pose refinement modules
* Experiment with hourglass or transformer decoders
* Extend to multi-person estimation using associative embedding
* Deploy trained model on webcam or video stream for real-time inference

---

## 📁 Repository

[GitHub Link](https://github.com/kanand-cfd/pose-estimation-project)


---
