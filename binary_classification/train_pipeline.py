import pandas as pd
import torch
from torch import nn
import os
from dataloader import BinaryClass
from model import Simple, Medium, AgeAlexNet
from torch.utils.data import DataLoader
import numpy as np

if __name__ == '__main__':
    path_main = os.path.split(os.getcwd())[0]
    df = pd.read_csv(path_main + '/files/train.txt', sep=' ', header=None)
    df.columns = ['name', 'class']
    #   df = df.sample(frac=1).reset_index(drop=True)
    # df_filtered = df[['image_name','old_young']]
    # print(df_filtered)
    path_to_images = '/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized/'  # make this dinamic and change the path
    # net = Simple()
    # net = Simple()
    #  net =  AgeAlexNet(pretrainded=False)
    net = Simple()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    dataset = BinaryClass(df, path_to_images)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=100)
    overall_loss = []
    for epoch in range(100):
        running_loss = 0.0

        for i, k in enumerate(dataset):
            X, y = k
            optimizer.zero_grad()
            predictions = net(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
                overall_loss.append(loss.item())
                torch.save(net.state_dict(), path_main + '/binary_classification/models/medium' + str(i) + '.pt')
                print('fin')
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')

        running_loss = 0.0
    print('finished_training' + str(loss.item()))
    print('mean_loss' + str(np.mean(overall_loss)))
