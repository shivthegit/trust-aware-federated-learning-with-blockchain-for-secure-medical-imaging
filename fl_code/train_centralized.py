import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    # Transforms (ResNet expects 224x224 + normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset
    data_dir = "../data/chest_xray/train"
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)

    # Freeze early layers (faster + stable on CPU)
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(device)

    # Class weights (handle imbalance)
    class_weights = torch.tensor([2.5, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    epochs = 6

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "../models/centralized_model.pth")

    print("Centralized ResNet model trained and saved")