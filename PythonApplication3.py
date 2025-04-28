import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Step 1: Load and preprocess the dataset (CIFAR-10 for demonstration)
def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
    ])
    full_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    return full_dataset
#shard the dataset into disjoint subsets 
def shard_dataset(full_dataset, num_shards=5):
    shard_size = len(full_dataset) // num_shards
    shards = [
        Subset(full_dataset, range(i * shard_size, (i + 1) * shard_size))
        for i in range(num_shards)
    ]
    return shards
#step 3 define what a simple CNN model is in the code 
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # Input: 3 channels (RGB)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(6 * 14 * 14, 10)  # Output: 10 classes (CIFAR-10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x
    # next we have to train the submodels on each shard 
    def train_submodels(shards, num_epochs=5):
    submodels = []
    for idx, shard in enumerate(shards):
        print(f"Training submodel on shard {idx + 1}/{len(shards)}")
        
        # Initialize model, optimizer, and loss function
        model = SimpleCNN()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create DataLoader for the current shard
        loader = DataLoader(shard, batch_size=64, shuffle=True)
        
        # Training loop
        for epoch in range(num_epochs):
            for X, y in loader:
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
        
        submodels.append(model)
    return submodels
# next we have to unlearn the data point by targeting the only affected shard
def unlearn_data(submodels, shards, shard_idx_to_unlearn, num_epochs=5):
    print(f"Unlearning data in shard {shard_idx_to_unlearn}")
    
    # Remove last 100 samples from the target shard (simulate data removal)
    original_shard = shards[shard_idx_to_unlearn]
    new_shard = Subset(original_shard, indices=range(len(original_shard) - 100))
    
    # Retrain only the affected submodel
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(new_shard, batch_size=64, shuffle=True)
    
    for epoch in range(num_epochs):
        for X, y in loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    
    # Update the submodels list
    submodels[shard_idx_to_unlearn] = model
    return submodels
# next is code that aggregate predictions from all the submodels 
def ensemble_predict(submodels, X):
    votes = torch.zeros(10)  # For CIFAR-10's 10 classes
    for model in submodels:
        with torch.no_grad():
            outputs = model(X)
            predicted_class = torch.argmax(outputs).item()
            votes[predicted_class] += 1
    return torch.argmax(votes).item()
# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    # Load and shard the dataset
    full_dataset = load_dataset()
    shards = shard_dataset(full_dataset, num_shards=5)
    
    # Train initial submodels
    submodels = train_submodels(shards)
    
    # Test the ensemble model
    test_image, test_label = full_dataset[0]  # Get a sample image
    prediction = ensemble_predict(submodels, test_image.unsqueeze(0))
    print(f"Initial prediction: {prediction}, True label: {test_label}")
    
    # Unlearn data from shard 2 and update models
    submodels = unlearn_data(submodels, shards, shard_idx_to_unlearn=2)
    
    # Test again after unlearning
    new_prediction = ensemble_predict(submodels, test_image.unsqueeze(0))
    print(f"Post-unlearning prediction: {new_prediction}, True label: {test_label}")