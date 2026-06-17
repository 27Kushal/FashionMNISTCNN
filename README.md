# FashionMNIST Image Classifier

A Convolutional Neural Network (CNN) built with PyTorch to classify clothing items 
from the FashionMNIST dataset across 10 categories.

Built while following [Daniel Bourke's PyTorch for Deep Learning](https://www.learnpytorch.io/) course.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 88.54% |
| Dataset | FashionMNIST (60k train / 10k test) |
| Framework | PyTorch |

## Model Architecture

- Conv Block 1: Conv2d → ReLU → Conv2d → ReLU → MaxPool2d
- Conv Block 2: Conv2d → ReLU → Conv2d → ReLU → MaxPool2d
- Classifier: Flatten → Linear

## Classes

T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot

## Usage

```python
# Download dataset (auto)
from torchvision import datasets, transforms
test_data = datasets.FashionMNIST(root="data", train=False, download=True)

# Load model
model.load_state_dict(torch.load("fashion_mnist_cnn.pth"))
model.eval()
```

## Dependencies

- torch
- torchvision
- torchmetrics
- mlxtend
- matplotlib
