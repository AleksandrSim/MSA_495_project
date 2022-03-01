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
import prepare_data_training


net = Simple()
model_conv = torchvision.models.resnet18(pretrained=False)
num_ftrs = model_conv.fc.in_features
for param in model_conv.parameters():
    param.requires_grad = False

model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 2))


states= torch.load('/Users/aleksandrsimonyan/Desktop/models/two_class439.pt')

model_conv.load_state_dict(states)

net = model_conv

path_main = os.path.split(os.getcwd())[0]
train, valid, weights = prepare_data_training.get_the_df(path_main + '/files/train.txt', class_2=True)
print(net)



classes = [0, 1]
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
path_to_images = '/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized/'  # make this dinamic and change the path

val_dataset = BinaryClass(valid, path_to_images,train=False)
val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=100)
# again no gradients needed
with torch.no_grad():
    for i, data in enumerate(val_dataset):
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
            if i %10 ==9:
                print('correct------->' + str(correct_pred))
                print('overall------->' + str(total_pred))



# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(accuracy)


