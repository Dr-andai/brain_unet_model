---
license: mit
tags:
  - image-segmentation
  - medical-imaging
  - pytorch
  - unet
datasets:
  - custom
inference: true
---

# Brain U-Net for MRI Segmentation

This repository contains a custom U-Net model trained to segment brain MRI images into white matter, gray matter, and cerebrospinal fluid (CSF) compartments.

- Format: PyTorch `.pth` state dict
- Input: Grayscale 256x256 MRI images
- Output: Segmentation map with 3 classes (WM, GM, CSF)
