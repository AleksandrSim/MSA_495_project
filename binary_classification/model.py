import torch.nn as nn
import torch.nn.functional as F
class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,12,5)
        self.conv3 = nn.Conv2d(12,20,5)

        self.maxPool = nn.MaxPool2d(2,2)

        self.Linear1 = nn.Linear(41772, 1000)
        self.activation = nn.Tanh()
        self.Linear = nn.Linear(1000, 500)
        self.sigmoid = nn.Sigmoid()
        self.Linear2 = nn.Linear(1000,500)
        self.Linear3 = nn.Linear(500, 100)
        self.out = nn.Linear(100,5)


    def forward(self, x):
        x = self.maxPool(self.conv1(x))
        x = self.maxPool(self.conv2(x))

        x = x.view(x.shape[0],-1)
        x = self.activation(self.Linear1(x))
        x = self.activation(self.Linear2(x))
        x = self.activation(self.Linear3(x))
        x = self.out(x)
        return x


class Medium(nn.Module):
    def __init__(self):
        super(Medium, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(111392, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 111392)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x


