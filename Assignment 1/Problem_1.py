import numpy as np
import cv2  # Only used for reading/writing images

# Read the image
image_color = cv2.imread('8bit_img.jpg')

# Function to convert an RGB image to Grayscale manually
def rgb_to_grayscale(image):

    # Initialize an empty grayscale image with the same height and width as the color image
    # The grayscale image has only one channel (intensity) instead of three (RGB).
    grayscale_image = np.zeros((image.shape[0], image.shape[1]))

    for i in range(image.shape[0]):  # Loop through image rows
        for j in range(image.shape[1]):  # Loop through image columns
            # Convert the pixel from RGB to grayscale using the following formula:
            # Grayscale value = 0.299 * R + 0.587 * G + 0.114 * B
            grayscale_image[i, j] = 0.299 * image[i, j, 2] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 0]
    return grayscale_image

# Apply the conversion function to the loaded color image
grayscale_image = rgb_to_grayscale(image_color)

# Save the grayscale image
cv2.imwrite('grayscale_image.png', grayscale_image)
