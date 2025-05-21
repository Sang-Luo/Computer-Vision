import numpy as np
import cv2
import matplotlib.pyplot as plt

# Helper function for convolution
def apply_convolution(image, kernel):
    output = np.zeros_like(image)
    pad = kernel.shape[0] // 2
    padded_image = np.pad(image, pad, mode='constant')

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = (kernel * padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]).sum()
    return output

# Gaussian Smoothing Function
def apply_gaussian_blur(image, kernel_size=5, sigma=1.4):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

# Sobel Edge Detection
def sobel_edge_detection(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal gradient
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)  # Gradient magnitude
    normalized = (magnitude / magnitude.max() * 255).astype(np.uint8)  # Normalize
    _, thresholded = cv2.threshold(normalized, 50, 255, cv2.THRESH_BINARY)  # Threshold
    return thresholded

# Prewitt Edge Detection
def prewitt_edge_detection(image):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Prewitt horizontal kernel
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # Prewitt vertical kernel
    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)  # Horizontal gradient
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)  # Vertical gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)  # Gradient magnitude
    normalized = (magnitude / magnitude.max() * 255).astype(np.uint8)  # Normalize
    _, thresholded = cv2.threshold(normalized, 50, 255, cv2.THRESH_BINARY)  # Threshold
    return thresholded

# Non-Maximum Suppression
def non_max_suppression(magnitude, direction):
    M, N = magnitude.shape
    output = np.zeros((M, N), dtype=np.uint8)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                output[i, j] = magnitude[i, j]
            else:
                output[i, j] = 0

    return output

# Hysteresis Thresholding
def hysteresis_thresholding(image, low_threshold, high_threshold):
    M, N = image.shape
    output = np.zeros((M, N), dtype=np.uint8)

    strong = 255
    weak = 75

    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))

    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if output[i, j] == weak:
                if strong in (output[i + 1, j - 1:j + 2].tolist() +
                              output[i, j - 1:j + 2].tolist() +
                              output[i - 1, j - 1:j + 2].tolist()):
                    output[i, j] = strong
                else:
                    output[i, j] = 0

    return output

# Canny Edge Detection
def canny_edge_detection(image, low_threshold, high_threshold):
    blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(grad_y, grad_x)

    nms_result = non_max_suppression(magnitude, direction)
    edges = hysteresis_thresholding(nms_result, low_threshold, high_threshold)
    return edges

# Add noise to image
def add_noise(image, noise_type="gaussian", salt_prob=0.05, gauss_std=20):
    noisy_image = image.copy()
    if noise_type == "gaussian":
        noise = np.random.normal(0, gauss_std, image.shape)
        noisy_image = np.clip(image + noise, 0, 255)
    elif noise_type == "salt_and_pepper":
        salt_pepper = np.random.rand(*image.shape)
        noisy_image[salt_pepper < salt_prob] = 0
        noisy_image[salt_pepper > 1 - salt_prob] = 255
    return noisy_image.astype(np.uint8)

# Main function to run the implementation
if __name__ == "__main__":
    # Read image using OpenCV
    image_path = "test.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Add noise to simulate diverse conditions
    noisy_image = add_noise(image, noise_type="gaussian", gauss_std=30)

    # Detect edges
    sobel_edges = sobel_edge_detection(noisy_image)
    prewitt_edges = prewitt_edge_detection(noisy_image)
    canny_edges = canny_edge_detection(noisy_image, low_threshold=50, high_threshold=100)

    # Save results
    cv2.imwrite("noisy_image.jpg", noisy_image)

    # Save results with debugging
    print(f"Sobel dtype: {sobel_edges.dtype}, min: {sobel_edges.min()}, max: {sobel_edges.max()}")
    cv2.imwrite("sobel_edges.jpg", sobel_edges)

    print(f"Prewitt dtype: {prewitt_edges.dtype}, min: {prewitt_edges.min()}, max: {prewitt_edges.max()}")
    cv2.imwrite("prewitt_edges.jpg", prewitt_edges)

    cv2.imwrite("canny_edges.jpg", canny_edges)

    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(231), plt.imshow(noisy_image, cmap='gray'), plt.title("Noisy Image")
    plt.subplot(232), plt.imshow(sobel_edges, cmap='gray'), plt.title("Sobel Edges")
    plt.subplot(233), plt.imshow(prewitt_edges, cmap='gray'), plt.title("Prewitt Edges")
    plt.subplot(234), plt.imshow(canny_edges, cmap='gray'), plt.title("Canny Edges")
    plt.show()
