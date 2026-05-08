import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ================= TRANSFORMS =================
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(   # 🔥 VERY IMPORTANT
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(   # 🔥 MUST MATCH TRAIN
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ================= LOAD DATA =================
def load_dataset(data_path):

    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")

    train_data = datasets.ImageFolder(
        train_dir,
        transform=get_train_transforms()
    )

    test_data = datasets.ImageFolder(
        test_dir,
        transform=get_test_transforms()
    )

    return train_data, test_data


# ================= DATALOADER =================
def get_loader(dataset, batch_size=16, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)