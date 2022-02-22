import pandas as pd
import torch
from torch import nn
import os
from dataloader import BinaryClass
from model import Simple
from torch.utils.data import DataLoader


if __name__ =='__main__':
    path_main =os.path.split(os.getcwd())[0]
    df = pd.read_csv(path_main + '/files/binary_class_data.csv')
    df_filtered = df[['image_name','age']]
    path_to_images = '/Users/aleksandrsimonyan/Desktop/CACD2000/' #make this dinamic and change the path obviously
    net = Simple()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    dataset = BinaryClass(df, path_to_images)
    dataset = torch.utils.data.DataLoader(dataset, batch_size= 50)
    for epoch in range(2):
        running_loss = 0.0
        for i,k in enumerate(dataset):
            print(i)

            X,y = k
            optimizer.zero_grad()
            predictions = net(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i %1000 == 999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
        running_loss =0.0

    print('finished_training' + str(loss.item()))














