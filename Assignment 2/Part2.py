import cv2
import numpy as np
import matplotlib.pyplot as plt


# Apply a low-pass filter (Gaussian blur) to make the image blurry
def low_pass_filter(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# Apply a high-pass filter by subtracting the low-pass filtered version and enhancing contrast
def high_pass_filter(image, kernel_size):
    low_pass = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    high_pass = cv2.subtract(image, low_pass)

    # Use absolute values to keep all high-frequency details
    high_pass = cv2.convertScaleAbs(high_pass)

    # Optionally, apply histogram equalization for better contrast
    for i in range(3):  # Apply to each color channel if in color
        high_pass[:, :, i] = cv2.equalizeHist(high_pass[:, :, i])

    return high_pass


# Create and save the hybrid image
def create_hybrid_image(imageA, imageB, low_pass_size, high_pass_size):
    # Ensure images match dimensions if necessary
    if imageA.shape != imageB.shape:
        imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

    # Apply low-pass and high-pass filters
    low_pass_image = low_pass_filter(imageA, low_pass_size)
    high_pass_image = high_pass_filter(imageB, high_pass_size)

    # Combine low-pass and high-pass images to create the hybrid image
    hybrid_image = cv2.addWeighted(low_pass_image, 0.5, high_pass_image, 0.5, 0)

    # Save and display each step
    cv2.imwrite('low_pass_image.png', low_pass_image)
    cv2.imwrite('high_pass_image.png', high_pass_image)
    cv2.imwrite('hybrid_image.png', hybrid_image)

    # Display the images
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(low_pass_image, cv2.COLOR_BGR2RGB))
    plt.title("Low-Pass Filtered Image (Blurry)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(high_pass_image, cv2.COLOR_BGR2RGB))
    plt.title("High-Pass Filtered Image (Sharp Details)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(hybrid_image, cv2.COLOR_BGR2RGB))
    plt.title("Hybrid Image")
    plt.axis('off')

    plt.show()


# Load images for hybrid creation in color
imageA = cv2.imread('imageA.png')  # Replace with actual path if needed
imageB = cv2.imread('imageB.png')  # Replace with actual path if needed

# Generate and save hybrid image with desired cutoff frequencies
# Adjust kernel sizes as needed for the desired effect
create_hybrid_image(imageA, imageB, low_pass_size=35, high_pass_size=15)
