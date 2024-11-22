import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # Convolution 1
        x = self.pool(x)  # Pooling 1
        x = self.relu(self.conv2(x))  # Convolution 2
        x = self.pool(x)  # Pooling 2
        x = x.view(-1, 16 * 7 * 7)  # Flatten
        x = self.relu(self.fc1(x))  # Fully Connected 1
        x = self.fc2(x)  # Fully Connected 2
        return x

# Initialize model, loss, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Visualize filters of the first convolutional layer
def visualize_filters(layer):
    filters = layer.weight.data.cpu().numpy()
    num_filters = filters.shape[0]
    fig, axs = plt.subplots(1, num_filters, figsize=(15, 15))
    for i in range(num_filters):
        axs[i].imshow(filters[i, 0], cmap='gray')
        axs[i].axis('off')
    plt.show()

print("Visualizing filters of the first convolutional layer...")
visualize_filters(model.conv1)

# Visualize feature maps for all layers
def visualize_feature_maps(image, model):
    model.eval()
    feature_maps = []
    with torch.no_grad():
        x = model.relu(model.conv1(image))  # After first conv
        feature_maps.append(x)
        x = model.pool(x)  # After first pool
        feature_maps.append(x)
        x = model.relu(model.conv2(x))  # After second conv
        feature_maps.append(x)
        x = model.pool(x)  # After second pool
        feature_maps.append(x)
    
    for i, fmap in enumerate(feature_maps):
        fmap_np = fmap.cpu().numpy()[0]
        num_features = fmap_np.shape[0]
        fig, axs = plt.subplots(1, num_features, figsize=(15, 15))
        fig.suptitle(f"Feature Maps - Layer {i+1}", fontsize=16)
        for j in range(num_features):
            axs[j].imshow(fmap_np[j], cmap='gray')
            axs[j].axis('off')
        plt.show()

# Select one image from the test set
test_image, _ = test_dataset[0]
test_image = test_image.unsqueeze(0).to(device)

print("Visualizing feature maps for all layers...")
visualize_feature_maps(test_image, model)



# pip install torch torchvision matplotlib numpy

