import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image as grayscale
def load_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

def save_image(image, filepath):
    cv2.imwrite(filepath, image)

# Gaussian smoothing kernel (3x3)
gaussian_kernel = (1 / 16) * np.array([[1, 2, 1],
                                       [2, 4, 2],
                                       [1, 2, 1]])

# Sobel Kernels
Kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

Ky = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])

# Apply convolution
def convolve(image, kernel):
    height, width = image.shape
    kernel_size = kernel.shape[0]
    offset = kernel_size // 2
    output = np.zeros_like(image)
    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            region = image[i - offset:i + offset + 1, j - offset:j + offset + 1]
            output[i, j] = np.sum(region * kernel)
    return output

# Calculate gradient magnitude
def gradient_magnitude(Gx, Gy):
    return np.sqrt(Gx**2 + Gy**2)

# Main function for Part (a)
def edge_detection(filepath):
    image = load_image(filepath)

    # Apply Gaussian smoothing
    smoothed_image = convolve(image, gaussian_kernel)

    # Compute Gx and Gy using Sobel kernels
    Gx = convolve(smoothed_image, Kx)
    Gy = convolve(smoothed_image, Ky)

    # Compute gradient magnitude
    magnitude = gradient_magnitude(Gx, Gy)
    magnitude = np.clip(magnitude, 0, 255)  # Clip values for display

    # Save and display the result
    save_image(magnitude, "edges_detected.png")
    plt.imshow(magnitude, cmap='gray')
    plt.title("Edge Detection Result")
    plt.axis('off')
    plt.show()

# Run the edge detection function
edge_detection('input_image.png')  # Replace 'input_image.png' with the actual image path
