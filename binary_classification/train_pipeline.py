import pandas as pd
import torch
from torch import nn
import os
from dataloader import BinaryClass
from model import Simple, Medium
from torch.utils.data import DataLoader


if __name__ =='__main__':
    path_main =os.path.split(os.getcwd())[0]
    df = pd.read_csv(path_main + '/files/cleaned_images.csv')
 #   df = df.sample(frac=1).reset_index(drop=True)
    #df_filtered = df[['image_name','old_young']]
    #print(df_filtered)
    path_to_images = '/Users/aleksandrsimonyan/Desktop/CACD2000/' #make this dinamic and change the path obviously
    #net = Simple()
    net = Simple()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    dataset = BinaryClass(df, path_to_images)
    dataset = torch.utils.data.DataLoader(dataset, batch_size= 1000)
    for epoch in range(4):
        running_loss = 0.0
        for i,k in enumerate(dataset):
            X,y = k
            optimizer.zero_grad()
            predictions = net(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i %100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')

        running_loss =0.0
    print('finished_training' + str(loss.item()))














