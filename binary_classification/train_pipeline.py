import pandas as pd
import torch
from torch import nn
import os
from dataloader import BinaryClass
from model import Simple, Medium, AgeAlexNet
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    path_main = os.path.split(os.getcwd())[0]
    df = pd.read_csv(path_main + '/files/train.txt', sep=' ', header=None)
    df.columns = ['name', 'class']
    classes_to_covert = list(df['class'])
    new= []
    for i in classes_to_covert:
        if i ==0 or i ==1:
            new.append(0)
        elif i==2 or i ==3:
            new.append(1)
        else:
            new.append(2)
    df['3_class'] = np.array(new)
    df = df.drop(['class'], axis =1)
    df.columns = ['name', 'class']
    train, valid = train_test_split(df, test_size=0.2, random_state=42  )
    train = train.reset_index(drop=True)
    valid = train.reset_index(drop=True)

    #   df = df.sample(frac=1).reset_index(drop=True)
    # df_filtered = df[['image_name','old_young']]
    # print(df_filtered)
    path_to_images = '/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized/'  # make this dinamic and change the path
    # net = Simple()
    # net = Simple()
    net =  AgeAlexNet(pretrained=False)
    model_conv = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model_conv.fc.in_features
    net = model_conv
    for param in model_conv.parameters():
        param.requires_grad = False
    model_conv.fc = nn.Linear(num_ftrs, 3)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    dataset = BinaryClass(train, path_to_images)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=100)
    val_dataset = BinaryClass(valid, path_to_images)
    val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=100)
    overall_loss = []
    for epoch in range(10):
        running_loss = 0.0

        for i, k in enumerate(dataset):
            X, y = k
            optimizer.zero_grad()
            predictions = net(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
                overall_loss.append(loss.item())
                torch.save(net.state_dict(), '/Users/aleksandrsimonyan/Desktop/models/resnet_3_class_model' + str(i) + '.pt')
                print('fin')

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in val_dataset:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('validation_accuracy------------->' + str(correct/total))




        running_loss = 0.0
    print('finished_training' + str(loss.item()))
    print('mean_loss' + str(np.mean(overall_loss)))
