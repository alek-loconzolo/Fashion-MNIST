import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


"""
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted.
"""

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Converts images to tensors
        transforms.Normalize(
            (0.5,), (0.5,)
        ),  # Normalize the images with mean=0.5 and std=0.5
    ]
)
# Use transforms to convert images to tensors and normalize them
batch_size = 64

"""
Load the dataset. 
"""

# Load the training dataset with the transform applied
trainset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

# Create the DataLoader for the training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Load the test dataset with the transform applied
testset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

# Create the DataLoader for the test set
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


"""
Design a neural network. 
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First convolutional layer (input channels = 1 because Fashion-MNIST images are grayscale, output channels = 32)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        # Second convolutional layer (input channels = 32, output channels = 64)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        # Max-pooling layer (kernel size = 2x2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # First fully connected layer (input features = 64*7*7, output features = 128)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)

        # Second fully connected layer (input features = 128, output features = 10 for the 10 classes)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Apply first convolutional layer, followed by ReLU activation and max-pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Apply second convolutional layer, followed by ReLU activation and max-pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output from the convolutional layers
        x = x.view(-1, 64 * 7 * 7)

        # Apply the first fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(x))

        # Apply the second fully connected layer (output logits for the 10 classes)
        x = self.fc2(x)

        return x


# Instantiate the model
net = Net()


"""

Making loss function and optimizer
"""

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

"""
Training the model
"""

num_epochs = 10
train_losses = []


for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Calculate and record the average loss for this epoch
    average_loss = running_loss / len(trainloader)
    train_losses.append(average_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

print("Finished Training")

torch.save(net.state_dict(), "fmnist.pth")  # Saves model file

"""
Evaluting the model
"""

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        inputs, labels = data  # Get inputs and labels from the test loader
        outputs = net(inputs)  # Forward pass: Get the model outputs

        # The class with the highest output value is the predicted class
        _, predicted = torch.max(outputs.data, 1)

        # Update the total count
        total += labels.size(0)

        # Update the correct count with the number of correct predictions
        correct += (predicted == labels).sum().item()

# Calculate and print the accuracy
accuracy = correct / total

print("Accuracy: ", correct / total)


"""
Generating images for correctly and incorrectly classified cases.
"""


# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")


# Fashion-MNIST class labels
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Find one incorrect and one correct classification
incorrect_image = None
incorrect_predicted_label = None
incorrect_true_label = None

correct_image = None
correct_predicted_label = None
correct_true_label = None

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        # Check for incorrect classification
        for i in range(len(labels)):
            if predicted[i] != labels[i] and incorrect_image is None:
                incorrect_image = images[i]
                incorrect_predicted_label = classes[predicted[i]]
                incorrect_true_label = classes[labels[i]]

            if predicted[i] == labels[i] and correct_image is None:
                correct_image = images[i]
                correct_predicted_label = classes[predicted[i]]
                correct_true_label = classes[labels[i]]

            # Stop once we have one incorrect and one correct image
            if incorrect_image is not None and correct_image is not None:
                break

    if incorrect_image is not None:
        # Plot incorrect classification
        plt.figure(figsize=(5, 5))
        imshow(incorrect_image)
        plt.title(
            f"Incorrectly Classified\nPredicted: {incorrect_predicted_label}, True: {incorrect_true_label}"
        )
        plt.show()

    if correct_image is not None:
        # Plot correct classification
        plt.figure(figsize=(5, 5))
        imshow(correct_image)
        plt.title(
            f"Correctly Classified\nPredicted: {correct_predicted_label}, True: {correct_true_label}"
        )
        plt.show()


# Plot the training loss over time
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker="o", linestyle="-", color="b")
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
