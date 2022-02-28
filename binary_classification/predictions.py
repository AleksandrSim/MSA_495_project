from model import *
import torch
from dataloader import *
import os
import pandas as pd
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
import torchvision
net = Simple()
model_conv = torchvision.models.resnet18(pretrained=False)
num_ftrs = model_conv.fc.in_features
for param in model_conv.parameters():
    param.requires_grad = False
model_conv.fc = nn.Linear(num_ftrs, 3)

net = model_conv

states= torch.load('/Users/aleksandrsimonyan/Desktop/models/resnet_3_class_model139.pt')

net.load_state_dict(states)



print(net)

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

# train, valid = train_test_split(df, test_size=0.2, random_state=42)

train = df.sample(frac=0.8, random_state=200)
valid = df.drop(train.index)
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)

classes = [0, 1, 2]
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
path_to_images = '/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized/'  # make this dinamic and change the path

val_dataset = BinaryClass(valid, path_to_images)
val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=100,train=False)
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


