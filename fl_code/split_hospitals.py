import numpy as np
from torch.utils.data import Subset

def split_into_hospitals(train_dataset, num_hospitals=3):
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)

    splits = np.array_split(indices, num_hospitals)

    hospital_datasets = []
    for i in range(num_hospitals):
        hospital_datasets.append(Subset(train_dataset, splits[i]))

    return hospital_datasets