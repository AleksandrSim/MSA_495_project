from model import *
import torch
from dataloader import *
import os
import pandas as pd
import torchvision
net = Simple()
model_conv = torchvision.models.resnet18(pretrained=True)
num_ftrs = model_conv.fc.in_features
for param in model_conv.parameters():
    param.requires_grad = False
model_conv.fc = nn.Linear(num_ftrs, 5)

net = model_conv

states= torch.load('/Users/aleksandrsimonyan/Desktop/models/resnet9.pt')

net.load_state_dict(states)



print(net)


classes = [0, 1, 2]
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    path_main = os.path.split(os.getcwd())[0]
    df = pd.read_csv(path_main + '/files/train.txt', sep=' ', header=None)
    df.columns = ['name', 'class']
    #   df = df.sample(frac=1).reset_index(drop=True)
    # df_filtered = df[['image_name','old_young']]
    # print(df_filtered)
    path_to_images = '/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized/'  # make this dinamic and change the path
    dataset = BinaryClass(df, path_to_images)
    dataset = torch.utils.data.DataLoader(dataset, batch_size= 100)
    for i, data in enumerate(dataset):
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


