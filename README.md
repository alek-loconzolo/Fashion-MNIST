# Fashion-MNIST Classification with PyTorch

This repository contains a PyTorch implementation for classifying images from the Fashion-MNIST dataset. The model is a Convolutional Neural Network (CNN) trained to recognize 10 different clothing categories. The project includes data preprocessing, model training, evaluation, and visualization of both correct and incorrect classifications.

## Requirements

Before running the code, ensure you have the following libraries installed:

- `numpy`
- `torch`
- `torchvision`
- `matplotlib`

You can install them using:

```bash
pip install numpy torch torchvision matplotlib
```

## Project Overview

### 1. Preprocessing the Dataset
The Fashion-MNIST dataset is automatically downloaded using `torchvision.datasets`. The images are converted to tensors and normalized with a mean of 0.5 and a standard deviation of 0.5.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

A batch size of 64 is used for both the training and test sets.

### 2. Neural Network Design
The neural network is a simple Convolutional Neural Network (CNN) with:
- Two convolutional layers with ReLU activation and max-pooling.
- Two fully connected layers to output the predictions for the 10 classes.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
```

### 3. Loss Function and Optimizer
The model uses Cross Entropy Loss as the loss function and the Adam optimizer with a learning rate of 0.001.

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
```

### 4. Training
The model is trained for 10 epochs, and the training loss is recorded at each epoch.

```python
for epoch in range(num_epochs):
    # Training loop
```

### 5. Evaluation
After training, the model is evaluated on the test set to compute its accuracy. Correct and incorrect classifications are identified and displayed.

### 6. Visualizing Results
The model visualizes one correctly classified image and one incorrectly classified image. It also plots the training loss over time.

### Example Output:

- **Correctly Classified Image**: The model correctly identifies the class of an image.
- **Incorrectly Classified Image**: The model's incorrect classification is shown along with the true label.
- **Training Loss Plot**: The loss is plotted over the 10 epochs to visualize the model's learning progress.

## How to Run the Code

1. Clone the repository and navigate to the folder.
2. Run the code using your preferred Python environment:
   ```bash
   python fashion_mnist.py
   ```

The code will automatically download the Fashion-MNIST dataset, train the CNN, evaluate its performance, and visualize the results.

## Results

After training, the model's accuracy on the test dataset is printed. Additionally, you will see the following visualizations:
- A plot of the training loss over time.
- Images showing correct and incorrect classifications, with the predicted and true labels.

## Model Saving

The trained model is saved to the file `fmnist.pth`, which can be reloaded for future use without retraining.

```python
torch.save(net.state_dict(), "fmnist.pth")
```

## License

This project is open-source and available under the MIT License.
