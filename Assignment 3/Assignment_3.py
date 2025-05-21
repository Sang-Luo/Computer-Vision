import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- Data Preprocessing ---
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset, test_dataset

# --- CNN Model ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- Feedforward Neural Network ---
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc_layers(x)

# --- Training Function ---
def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Val Accuracy: {val_correct / len(val_loader.dataset):.4f}")

# --- Evaluation Function ---
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# --- Adding Noise ---
def add_noise(images, noise_type="gaussian", param=0.1):
    noisy_images = []
    for img in images:
        img = img.numpy()
        if noise_type == "gaussian":
            noise = np.random.normal(0, param, img.shape)
            noisy_img = img + noise
        elif noise_type == "salt_pepper":
            noisy_img = img.copy()
            num_salt = int(param * img.size)
            coords = [np.random.randint(0, i, num_salt) for i in img.shape]
            noisy_img[tuple(coords)] = 1
            num_pepper = int(param * img.size)
            coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
            noisy_img[tuple(coords)] = 0
        noisy_images.append(np.clip(noisy_img, 0, 1))
    return torch.tensor(np.array(noisy_images))

# Save images as PNG
def save_image(tensor, filename):
    img = tensor.squeeze().numpy()  # Remove batch/channel dimension and convert to numpy
    img = (img * 255).astype(np.uint8)  # Convert to 0-255 scale
    Image.fromarray(img).save(filename)

# --- Main Function ---
# --- Main Function ---
def main():
    train_loader, val_loader, test_loader, train_dataset, test_dataset = load_data()

    print("Training CNN...")
    cnn_model = CNN()
    train_model(cnn_model, train_loader, val_loader)
    evaluate_model(cnn_model, test_loader)

    print("Training Feedforward Neural Network...")
    fnn_model = FNN()
    train_model(fnn_model, train_loader, val_loader)
    evaluate_model(fnn_model, test_loader)

    print("Training CNN with reduced data...")
    # Task 3: Training on 50% and 5% of data
    half_train_size = len(train_dataset) // 2
    small_train_size = len(train_dataset) // 20

    half_train_loader = DataLoader(torch.utils.data.Subset(train_dataset, range(half_train_size)), batch_size=64,
                                   shuffle=True)
    small_train_loader = DataLoader(torch.utils.data.Subset(train_dataset, range(small_train_size)), batch_size=64,
                                    shuffle=True)

    train_model(cnn_model, half_train_loader, val_loader)
    evaluate_model(cnn_model, test_loader)

    train_model(cnn_model, small_train_loader, val_loader)
    evaluate_model(cnn_model, test_loader)

    print("Testing robustness to noise...")
    # Task 4: Add noise and evaluate
    test_images = [test_dataset[i][0] for i in range(2)]  # Fetch two test images

    # Save original images
    for i, img in enumerate(test_images):
        save_image(img, f"original_{i}.png")

    noisy_gaussian = add_noise(test_images, noise_type="gaussian", param=0.2)
    noisy_salt_pepper = add_noise(test_images, noise_type="salt_pepper", param=0.05)

    # Save noisy images
    for i, noisy_img in enumerate(noisy_gaussian):
        save_image(noisy_img, f"gaussian_{i}.png")

    for i, noisy_img in enumerate(noisy_salt_pepper):
        save_image(noisy_img, f"saltpepper_{i}.png")

    # Display noisy images (optional)
    for i, noisy_img in enumerate(noisy_gaussian):
        plt.subplot(1, 4, i + 1)
        plt.imshow(noisy_img.squeeze(), cmap="gray")
        plt.title("Gaussian Noise")

    for i, noisy_img in enumerate(noisy_salt_pepper):
        plt.subplot(1, 4, i + 3)
        plt.imshow(noisy_img.squeeze(), cmap="gray")
        plt.title("Salt & Pepper Noise")
    plt.show()

    # Check if the CNN is fooled by noise
    print("Checking if CNN is fooled by noise...")

    # Predict original images
    original_predictions = []
    for img in test_images:
        cnn_model.eval()
        with torch.no_grad():
            img = img.unsqueeze(0)  # Add batch dimension
            output = cnn_model(img)
            predicted_label = torch.argmax(output, dim=1).item()
            original_predictions.append(predicted_label)

    print(f"Original Predictions: {original_predictions}")

    # Predict Gaussian noisy images
    noisy_gaussian_predictions = []
    for noisy_img in noisy_gaussian:
        cnn_model.eval()
        with torch.no_grad():
            noisy_img = noisy_img.unsqueeze(0).to(torch.float32)  # Add batch dimension and convert to float32
            output = cnn_model(noisy_img)
            predicted_label = torch.argmax(output, dim=1).item()
            noisy_gaussian_predictions.append(predicted_label)

    # Predict Salt & Pepper noisy images
    noisy_salt_pepper_predictions = []
    for noisy_img in noisy_salt_pepper:
        cnn_model.eval()
        with torch.no_grad():
            noisy_img = noisy_img.unsqueeze(0).to(torch.float32)  # Add batch dimension and convert to float32
            output = cnn_model(noisy_img)
            predicted_label = torch.argmax(output, dim=1).item()
            noisy_salt_pepper_predictions.append(predicted_label)

    print(f"Salt & Pepper Noise Predictions: {noisy_salt_pepper_predictions}")

    # Compare predictions and determine if CNN is fooled
    for i in range(2):
        print(f"Image {i + 1}:")
        print(f"  Original Prediction: {original_predictions[i]}")
        print(f"  Gaussian Noise Prediction: {noisy_gaussian_predictions[i]}")
        print(f"  Salt & Pepper Noise Prediction: {noisy_salt_pepper_predictions[i]}")

        if original_predictions[i] != noisy_gaussian_predictions[i]:
            print("  CNN was fooled by Gaussian Noise!")
        else:
            print("  CNN was not fooled by Gaussian Noise.")

        if original_predictions[i] != noisy_salt_pepper_predictions[i]:
            print("  CNN was fooled by Salt & Pepper Noise!")
        else:
            print("  CNN was not fooled by Salt & Pepper Noise.")


if __name__ == "__main__":
    main()
