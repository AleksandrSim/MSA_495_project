from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

# i added stuff to debug direclty here, can remove later
import os
import pandas as pd
import numpy as np
from global_variables import *


class BinaryClass(Dataset):
    def __init__(self, dataset, path_to_images, train=True):
        self.dataset = dataset
        self.path_to_images = path_to_images
        self.train = train

        transform = {"train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6673, 0.4973, 0.4194),
                                                     (0.2420, 0.2107, 0.2029))]),
            "valid": transforms.Compose([transforms.ToTensor()])}

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        y = torch.tensor(int(self.dataset['class'].iloc[index])).long()

        X = Image.open(self.path_to_images + self.dataset['name'][index])

        if self.train:
            X = self.transform["train"](X)
        else:
            X = self.transform["valid"](X)

        return X, y


if __name__ == '__main__':

    # can remove this all later
    path_main = os.path.split(os.getcwd())[0]
    df = pd.read_csv(path_main + '/files/train.txt', sep=' ', header=None)
    df.columns = ['name', 'class']
    classes_to_covert = list(df['class'])
    new = []
    for i in classes_to_covert:
        if i == 0 or i == 1:
            new.append(0)
        elif i == 2 or i == 3:
            new.append(1)
        else:
            new.append(2)
    df['3_class'] = np.array(new)
    df = df.drop(['class'], axis=1)
    df.columns = ['name', 'class']

    train = df.sample(frac=0.8, random_state=200)
    valid = df.drop(train.index)

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)

    train_dataset = BinaryClass(train, clean_images_output_path, train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100)

    val_dataset = BinaryClass(valid, clean_images_output_path, train=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100)
