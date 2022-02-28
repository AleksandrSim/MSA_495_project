from torch.utils.data import Dataset
import torchvision.transforms as transforms
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

        # transform = {"train": transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor()]),
        #     "valid": transforms.Compose([transforms.ToTensor()])}
        #
        # self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        y = torch.tensor(int(self.dataset['class'].iloc[index])).long()

        X = Image.open(self.path_to_images + self.dataset['name'][index])

        # if self.train == True:
        #     X = self.transform["train"](X)
        # else:
        #     X = self.transform["valid"](X)

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

    train = df.sample(frac=0.8,random_state=200)
    valid = df.drop(train.index)

    train_dataset = BinaryClass(train, original_images_path, train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100)

    val_dataset = BinaryClass(valid, original_images_path, train=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100)


