# 📘 Enhanced Human

This repository contains the code developped by Neurobus for the Enhanced Human project. It allows pre-training and the real-time pipeline for inference of Akida and CPU models.

---

## 🧭 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Basic Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Results](#results)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
---

## ✨ Features


- Pre-training and evaluation of Akida and CPU/GPU models, saving models and CSP objects.  
- Dataset pre-processing allowing for creating datasets of any possible combination of classes and for n-way classification
- Real-time inference pipeline for Akida and CPU/GPU models.
    - Reads input EEG data from usb port
    - Pre-process data (reshaping, filtering, quantization if needed)
    - Inference on CPU or Akida Chip
    - Sends POST request to server
- Modular EEG prediction strategy  
- Example of energy consumption calculation 

---

## 🛠️ Installation

Before proceeding to installation make sure to python and tensorflow versions that respect:
```
3.9 <= python <= 3.11
tensorflow <= 2.15
Keras <= 2.15
```
This will ensure compatibility with the BrainChip MetaTF frameworks, which can be installed using:
```
pip install akida==2.12.0
pip install cnn2snn==2.12.0
pip install akida-models==1.6.3
```

After this, you can continue installing dependencies as you see fit, or use the requirements.txt file:
```bash
# Install required packages
pip install -r requirements.txt
```

##  🚀 Basic usage

```
# Kernel size should be 3, 5, 7 and symmetric
# strides should be 1, 2 or 3 symmetric (input) only 1,2 intermediary
# Input dimension cannot be smaller than 5
# Max input size is 256
# convolution stride 2 only supported for 3x3```