# MNIST Classification using PyTorch

This project implements a Convolutional Neural Network (CNN) and a Feedforward Neural Network (FNN) to classify handwritten digits from the MNIST dataset. The project evaluates:
1. The performance of CNNs and FNNs.
2. The impact of reduced training data on CNN performance.
3. The validity of CNN against noisy inputs.

---

## **Features**
1. **CNN Architecture**:
   - 4 convolutional layers, 3 max-pooling layers, and 1 fully connected layer.
   - Trained to achieve high accuracy on the MNIST dataset.

2. **FNN Architecture**:
   - Two hidden layers (128 and 64 neurons) and an output layer.
   - Compared with the CNN for performance.

3. **Reduced Training Data**:
   - CNN is trained with 50% and 5% of the training dataset.
   - Observes the effect on accuracy.

4. **Noise Robustness**:
   - Tests CNN's robustness to Gaussian and salt-and-pepper noise added to test images.

---

## **Requirements**
- Python 3.7 or higher
- PyTorch
- torchvision
- matplotlib
- numpy

Install dependencies using:
```bash
pip install torch torchvision matplotlib numpy
```

---

## **Expected Output**

The program performs the following tasks:

1. Training the CNN:

Training and validation metrics (loss and accuracy) for 10 epochs.
Test accuracy after training (e.g., ~98.7%).

2. Training the FNN:

Similar metrics for the feedforward neural network.
Test accuracy (e.g., ~96.9%).

3. Training CNN on Reduced Data:

Test accuracy with:
: 50% of training data (e.g., ~94%).
: 5% of training data (e.g., ~87%).

4. Robustness to Noisy Images:

Test accuracy with Gaussian and salt-and-pepper noise.
Visualizations of noisy images and their effect on CNN classification.

---

## **File Structure**

- Assignment_3.py: Main Python file implementing the tasks.
- readme.md: Instructions for setup and usage.
- report.md: Analysis and answers to assignment tasks.

---