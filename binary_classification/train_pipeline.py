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

if __name__ == '__main__':
    # original_images_path = '/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized/'
    original_images_path = "/Users/joshcheema/Box Sync/MSAI/MSAI 495 DL/project_data/cross_age_dataset_cleaned_and_resized/"
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

    train, valid = train_test_split(df, test_size=0.2, random_state=42)
    train = df.sample(frac=0.8,random_state=200)
    valid = df.drop(train.index)

    # # train = train.reset_index(drop=True)
    # # valid = train.reset_index(drop=True)
    #
    # #   df = df.sample(frac=1).reset_index(drop=True)
    # # df_filtered = df[['image_name','old_young']]
    # # print(df_filtered)  # make this dinamic and change the path
    # # net = Simple()
    # # net = Simple()
    # model_conv = torchvision.models.vgg16_bn(pretrained=True)
    # #num_ftrs = model_conv.classifier.
    # for param in model_conv.parameters():
    #     param.requires_grad = False
    # model_conv.classifier = nn.Sequential(
    #         nn.Linear(25088 , 512),
    #         nn.BatchNorm1d(512),
    #         nn.Dropout(0.2),
    #         nn.Linear(512 , 256),
    #         nn.Linear(256 , 3)
    #     )
    # net = model_conv
    #
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # dataset = BinaryClass(train, original_images_path)
    # dataset = torch.utils.data.DataLoader(dataset, batch_size=100)
    # val_dataset = BinaryClass(valid, original_images_path)
    # val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=100)
    # overall_loss = []
    #
    # for epoch in range(10):
    #     running_loss = 0.0
    #
    #     for i, k in enumerate(dataset):
    #         X, y = k
    #         optimizer.zero_grad()
    #         predictions = net(X)
    #         loss = criterion(predictions, y)
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item()
    #         if i % 10 == 9:
    #             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
    #             overall_loss.append(loss.item())
    #             torch.save(net.state_dict(), class_model_path_Alex + str(i) + '.pt')
    #             print('fin')
    #
    #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
    #     correct = 0
    #     total = 0
    #     # since we're not training, we don't need to calculate the gradients for our outputs
    #     with torch.no_grad():
    #         for data in val_dataset:
    #             images, labels = data
    #             # calculate outputs by running images through the network
    #             outputs = net(images)
    #             # the class with the highest energy is what we choose as prediction
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #     print('validation_accuracy------------->' + str(correct / total))
    #
    #     running_loss = 0.0
    # print('finished_training' + str(loss.item()))
    # print('mean_loss' + str(np.mean(overall_loss)))
