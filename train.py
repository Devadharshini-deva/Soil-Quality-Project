import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.cluster import KMeans
import numpy as np

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_dir = "./SoilDatasets/train_data"  # Update path to dataset location
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load Pretrained Model (ResNet18)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

# Modify model: Classification + pH Regression
class SoilClassificationWithPH(nn.Module):
    def __init__(self, base_model, num_classes=4):
        super(SoilClassificationWithPH, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC layer
        self.fc_class = nn.Linear(num_ftrs, num_classes)  # Soil type classification
        self.fc_regress = nn.Linear(num_ftrs, 1)  # pH level regression

    def forward(self, x):
        features = self.base_model(x)  # Extract features
        features = torch.flatten(features, 1)  # Flatten output
        
        class_output = self.fc_class(features)  # Soil type prediction
        ph_output = self.fc_regress(features)  # pH level prediction
        
        return class_output, ph_output

# Initialize model
model = SoilClassificationWithPH(model).to(device)

# Define loss functions and optimizer
criterion_class = nn.CrossEntropyLoss()  # Classification Loss
criterion_regress = nn.MSELoss()  # Regression Loss (for pH values)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Train Model with pH Regression
# -------------------------------
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        class_outputs, ph_outputs = model(images)
        
        # Generate synthetic pH values for training
        # Assume an approximate mapping of pH ranges (Modify if real data is available)
        ph_labels = torch.tensor([5.5 + i for i in labels], dtype=torch.float32).to(device)
        
        loss_class = criterion_class(class_outputs, labels)
        loss_regress = criterion_regress(ph_outputs.squeeze(), ph_labels)
        loss = loss_class + loss_regress
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = class_outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Save trained model
torch.save(model.state_dict(), "soil_ph_classification_model.pth")
print("Model training complete and saved!")

# -------------------------------
# Extract Features for K-Means Clustering
# -------------------------------
features = []
with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(device)
        encoded_features, _ = model(images)  
        encoded_features = encoded_features.cpu().numpy()
        features.append(encoded_features)

features = np.vstack(features)

# -------------------------------
# Apply K-Means Clustering for pH Grouping
# -------------------------------
num_clusters = 3  # Acidic, Neutral, Alkaline
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

# -------------------------------
# Assign pH Labels to Clusters
# -------------------------------
ph_labels = {
    0: "Acidic (pH < 6)",
    1: "Neutral (pH ~ 7)",
    2: "Alkaline (pH > 7.5)"
}

for i, label in enumerate(clusters[:10]):
    print(f"Image {i}: Cluster {label} → {ph_labels[label]}")
