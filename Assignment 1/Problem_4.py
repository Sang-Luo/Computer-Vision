import numpy as np
import cv2

# Read the noisy images generated from Problems 2 and 3
# noise_std_50.png contains white Gaussian noise with std deviation of 50
# sp_noise_30.png contains Salt and Pepper noise at 30%

# Read the noisy images, as grayscale
noise_std_50_image = cv2.imread('noise_std_50.png', cv2.IMREAD_GRAYSCALE)
sp_noise_30_image = cv2.imread('sp_noise_30.png', cv2.IMREAD_GRAYSCALE)

# Box Filter - This filter averages all pixel values within a neighborhood defined by the kernel size
def box_filter(image, kernel_size=3):

    # Pad the image to handle border pixels
    # THe padding adds zeros around the image, so the filtering can be applied uniformly even at the edges
    padded_image = np.pad(image, kernel_size // 2, mode='constant', constant_values=0)

    # Initialize an image for the result, with the same size as the original image
    filtered_image = np.zeros_like(image)

    # Loop through every pixel in the image
    for i in range(image.shape[0]): #Rows
        for j in range(image.shape[1]): # Columns
            # For each pixel, compute the mean of the neighborhood defined by the kernel size
            # np.mean() computes the arithmetic mean of the pixels inside the kernel window
            filtered_image[i, j] = np.mean(padded_image[i:i + kernel_size, j:j + kernel_size])

    return filtered_image

# Median Filter - This filter replaces each pixel with the median value of its surrounding pixels
def median_filter(image, kernel_size=3):

    # Very similar to Box Filter
    padded_image = np.pad(image, kernel_size // 2, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # For each pixel, compute the median value of the neighborhood defined by the kernel
            # The median is the middle value when all neighboring pixels are sorted
            filtered_image[i, j] = np.median(padded_image[i:i + kernel_size, j:j + kernel_size])

    return filtered_image

# Gaussian Filter - This filter applies a weighted average where the weights follow a Gaussian distribution
def gaussian_kernel(size, sigma=1):

    # The Gaussian function: G(x, y) = (1 / 2πσ²) * exp(-[(x² + y²) / 2σ²])
    # Where σ is the standard deviation (controls the spread of the Gaussian)
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )

    # Normalize the kernel so that the sum of all weights is 1
    return kernel / np.sum(kernel)

# Gaussian Filter function applies the kernel to the image
def gaussian_filter(image, kernel_size=3, sigma=1):

    # Generate the Gaussian kernel based on the size and sigma
    kernel = gaussian_kernel(kernel_size, sigma)

    padded_image = np.pad(image, kernel_size // 2, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Apply the Gaussian kernel to the neighborhood by taking a weighted sum
            # The kernel weighs the center pixel more heavily and the surrounding pixels less
            filtered_image[i, j] = np.sum(padded_image[i:i + kernel_size, j:j + kernel_size] * kernel)

    return filtered_image

# Apply filters to the Gaussian noise image (noise_std_50.png)
box_filtered_gaussian = box_filter(noise_std_50_image)
median_filtered_gaussian = median_filter(noise_std_50_image)
gaussian_filtered_gaussian = gaussian_filter(noise_std_50_image, kernel_size=3, sigma=1)

# Save the filtered images for Gaussian noise
cv2.imwrite('box_filtered_gaussian.png', box_filtered_gaussian)
cv2.imwrite('median_filtered_gaussian.png', median_filtered_gaussian)
cv2.imwrite('gaussian_filtered_gaussian.png', gaussian_filtered_gaussian)

# Apply filters to the Salt and Pepper noise image (sp_noise_30.png)
box_filtered_sp = box_filter(sp_noise_30_image)
median_filtered_sp = median_filter(sp_noise_30_image)
gaussian_filtered_sp = gaussian_filter(sp_noise_30_image, kernel_size=3, sigma=1)

# Save the filtered images for Salt and Pepper noise
cv2.imwrite('box_filtered_sp.png', box_filtered_sp)
cv2.imwrite('median_filtered_sp.png', median_filtered_sp)
cv2.imwrite('gaussian_filtered_sp.png', gaussian_filtered_sp)
