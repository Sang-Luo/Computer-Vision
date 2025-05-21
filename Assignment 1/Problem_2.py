import numpy as np
import cv2

# Ensure image is read in grayscale mode
grayscale_image = cv2.imread('grayscale_image.png', cv2.IMREAD_GRAYSCALE)

# Function to add Gaussian noise to an image
# Gaussian noise is random noise that follows a normal distribution (bell curve).
# std_dev is the standard deviation of the Gaussian distribution, which controls how much noise is added.
def add_gaussian_noise(image, std_dev):

    # The mean of the Gaussian distribution is set to 0, so the noise has no overall bias
    mean = 0

    # Generate noise with float32 to reduce memory usage
    # np.random.normal() generates a NumPy array of the same shape as the input image.
    # The values will follow a normal distribution , since we set the mean to 0
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)

    # Add noise to the image and clip the values to stay between 0 and 255
    # We convert the image to float32 to ensure the calculations handle decimal points.
    noisy_image = image.astype(np.float32) + noise

    # np.clip() ensures that values below 0 become 0 and values above 255 become 255.
    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert the result back to uint8 (standard for images)
    return noisy_image.astype(np.uint8)


# Apply noise with different standard deviations
noise_1 = add_gaussian_noise(grayscale_image, 1)
noise_10 = add_gaussian_noise(grayscale_image, 10)
noise_30 = add_gaussian_noise(grayscale_image, 30)
noise_50 = add_gaussian_noise(grayscale_image, 50)

# Save the noisy images
cv2.imwrite('noise_std_1.png', noise_1)
cv2.imwrite('noise_std_10.png', noise_10)
cv2.imwrite('noise_std_30.png', noise_30)
cv2.imwrite('noise_std_50.png', noise_50)
