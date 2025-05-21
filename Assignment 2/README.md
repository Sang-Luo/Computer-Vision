# **Assignment 2: Image Processing with Edge Detection and Hybrid Image Creation** #

### CSCI 4625: Computer Vision (Fall 2024) ###

___


This project consists of two parts: edge detection without the use of computer vision libraries (except for image 
reading and writing) and hybrid image creation using low and high-pass filtering. I am working on making more detailed 
README files, as I realized my last assignment I did was not as detailed with its README.

# Table of contents #
1. [Part A: Edge Detection](#part_a)
2. [Part B: Hybrid Image Creation](#part_b)
3. [Requirements](#requirements)
4. [Usage](#usage)
   1. [Running Part A](#part_a_htd)
   2. [Running Part B](#part_b_htd)


___

## Part A: Edge Detection  <a name="part_a"></a> ##

In Part A, the objective is to detect edges in a grayscale image using a custom implementation without computer vision 
libraries for processing (other than image I/O). The steps are as follows:

1. Apply a Gaussian smoothing filter to reduce noise.
2. Use Sobel operators to compute the horizontal and vertical gradients.
3. Calculate the gradient magnitude to highlight edges in the image.

**Results -**
- The output image, edges_detected.png, displays detected edges in the grayscale image.

___

## Part B: Hybrid Image Creation  <a name="part_b"></a> ##

In Part B, a hybrid image is created by combining the low-frequency content from one image (imageA.png) and the 
high-frequency content from another image (imageB.png).

1. **Low-Pass Filter:** A Gaussian blur is applied to imageA to retain only the low-frequency details.
2. **High-Pass Filter:** The high-frequency details of imageB are isolated by subtracting a blurred version of imageB.
3. **Hybrid Image:** The low-pass and high-pass filtered images are combined.

**Results -**
- low_pass_image.png: Blurry version of imageA.
- high_pass_image.png: Detailed version of imageB with enhanced high frequencies.
- hybrid_image.png: Combination of low-pass and high-pass images, resulting in a hybrid image that appears different
based on viewing distance.

___

## Requirements <a name="requirements"></a>

- Python 3.8
- numpy for numerical operations
- opencv-python (cv2) for image processing
- matplotlib for image display
- Pycharm (Optional)

___

## Usage <a name="usage"></a>

### Running Part A <a name="part_a_htd"></a>

1. Ensure that the file input_image.png is in the root folder.
2. Run the program.
3. Program will display image, and will save resulting image as edges_detected.png.

### Running Part B <a name="part_b_htd"></a>

1. Ensure that the file imageA.png and imageB.png is in the root folder.
2. Run the program.
3. Program will display image, and will save resulting images as high_pass_image.png, hybrid_image.png, and 
low_pass_image.png.

___