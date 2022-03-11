
import torch
import yaml
from torch import nn

from dataloader import BinaryClass
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser
import prepare_data_training
import os

from global_variables import *

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = ArgumentParser()
    parser.add_argument('--config', default='../config_files/config.yaml', help='Config .yaml file to use for training')

    # To read the data directory from the argument given
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)

    # To read the data directory from the argument given

    #generate_dir_if_not_exists( + classification_class_model_path)
    train, valid, weights = prepare_data_training.get_the_df(os.path.split(os.getcwd())[0] + '/files/train.txt', class_2=True)
    net = prepare_data_training.get_the_model()
    weights = torch.Tensor(
        [0.9410346097201767, 0.7480762150220913, 0.7339930044182621, 0.7580817378497791, 1.8188144329896907])
    weights_to_class = torch.Tensor([0.3676872403338698, 0.6323127596661302])
    criterion = nn.CrossEntropyLoss(weight=weights_to_class)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    dataset = BinaryClass(train, PATH_TO_IMAGES,train=True)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=100)
    val_dataset = BinaryClass(valid, PATH_TO_IMAGES,train=False)
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
                #torch.save(net.state_dict(), user_path + classification_class_model_path + 'two_class' + str(i) + '.pth')
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
        print('validation_accuracy------------->' + str(correct / total))

        running_loss = 0.0
    print('finished_training -> ' + str(loss.item()))
    print('mean_loss - > ' + str(np.mean(overall_loss)))
