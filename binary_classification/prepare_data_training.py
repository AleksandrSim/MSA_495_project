import os
import pandas as pd
import torch
from torch import nn
import os
from global_variables import *
from dataloader import BinaryClass
from model import Simple, Medium, AgeAlexNet
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split

NUM_CLASSES = 2


def get_the_df(path, class_3=False, class_2=False):
    df = pd.read_csv(path, sep=' ', header=None)
    df.columns = ['name', 'class']
    classes_to_covert = list(df['class'])
    if class_3:
        new = []
        for i in classes_to_covert:
            if i == 0 or i == 1:
                new.append(0)
            elif i == 2 or i == 3:
                new.append(1)
            else:
                new.append(2)
        df['3_class'] = np.array(new)
        df['3_class'] = np.array(new)
        df = df.drop(['class'], axis=1)
    if class_2:
        df = df[df['class'] != 2]
        df = df[df['class'] != 3]
        new = []
        for i in classes_to_covert:
            if i == 0 or i == 1:
                new.append(0)
            elif i == 4:
                new.append(1)
        df = df.reset_index(drop=True)

        df['3_class'] = np.array(new)
        df = df.drop(['class'], axis=1)

    df.columns = ['name', 'class']
    train, valid = train_test_split(df, test_size=0.2, random_state=42)
    train = train.reset_index(drop=True)
    valid = train.reset_index(drop=True)
    weights = train.groupby(['class']).count().reset_index()
    su = sum(weights['name'])
    raw = list(weights['name'])
    weights_final = [1 - (i / su) for i in raw]
    print(weights_final)
    return train, valid, weights_final


def get_the_model(pretrained=True):
    model_conv = torchvision.models.resnet18(pretrained=pretrained)
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                  nn.ReLU(),
                                  nn.Dropout(0.2),
                                  nn.Linear(512, NUM_CLASSES))
    #   nn.Softmax(dim=1)))
    net = model_conv
    return net


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print('mean----------> ' + str(mean))
    print('std----------> ' + str(std))

    return mean, std
