# Final Project: Edge Detection Techniques Analysis

### CSCI 4625: Computer Vision (Fall 2024)

---

# Table of contents #

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [Files](#files)
6. [Results](#results)


---

## Overview  <a name="overview"></a> 

This project implements and compares three popular-edge detection algorithms:
- Sobel
- Prewitt
- Canny

The objective is to analyze their performance in detecting edges from noisy images and evaluate their strengths and 
limitations under different conditions. The project also explores how these algorithms can be applied in real-world 
scenarios like object recognition and image segmentation.

---

## Features <a name="features"></a> 

1. Implementation of Sobel, Prewitt, and Canny edge detection.
2. Support for adding Gaussian noise to simulate diverse conditions.
3. Outputs visual results for edge detection on noisy images.
4. Provides debugging information to understand pixel intensity ranges.

---

## Requirements <a name="requirements"></a>

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib

---

## Usage <a name="usage"></a>

1. Ensure test.jpg file is in directory, if you want you can switch out the image file with another, just rename it to 
test.jpg.
2. Run Project.py
3. The program outputs will be saved in the working directory, and display a plots for comparison.

---

## Files

- **test.jpg**: Test image file used for testing.
- **Project.py**: Main script that implements the edge detection methods (Sobel, Prewitt, Canny) and adds noise to test images.
- **outputs/**: Directory for storing generated edge-detected images (e.g., `sobel_edges.jpg`, `prewitt_edges.jpg`, `canny_edges.jpg`).

---

## Results

The project produces:
- Visual outputs comparing Sobel, Prewitt, and Canny edge detection.
- Debugging information showing intensity ranges and pixel data of results.

---
