import numpy as np
import cv2

# Read the image in grayscale mode
grayscale_image = cv2.imread('grayscale_image.png', cv2.IMREAD_GRAYSCALE)


# Function to add Salt and Pepper noise to an image
# Salt and Pepper noise introduces random white (salt) and black (pepper) pixels to an image.
# 'prob' defines the probability of a pixel being affected by noise.
def add_salt_and_pepper_noise(image, prob):

    # Make a copy of the original image so we don't modify it directly
    noisy_image = np.copy(image)

    # Number of 'salt' (white) pixels to be added
    # 'prob' determines the percentage of the image affected by noise
    # We multiply by 0.5 since half of the noise is 'salt' (white) and the other half is 'pepper' (black)
    num_salt = np.ceil(prob * image.size * 0.5)

    # Number of 'pepper' (black) pixels to be added
    num_pepper = np.ceil(prob * image.size * 0.5)

    # Add salt (white) noise
    # Randomly choose 'num_salt' pixel coordinates where the noise will be added.
    # np.random.randint generates random coordinates. The range of the random integers is the size of the image (height, width).
    # The image.shape gives us the dimensions of the image
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]

    # Set the chosen coordinates to 255 (white), which represents 'salt' in grayscale.
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper (black) noise
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]

    # Set the chosen coordinates to 0 (black), which represents 'pepper' in grayscale.
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image


# Apply salt and pepper noise
sp_noise_10 = add_salt_and_pepper_noise(grayscale_image, 0.1)
sp_noise_30 = add_salt_and_pepper_noise(grayscale_image, 0.3)

# Save the noisy images
cv2.imwrite('sp_noise_10.png', sp_noise_10)
cv2.imwrite('sp_noise_30.png', sp_noise_30)
