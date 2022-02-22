import torch.nn as nn
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
        self.out = nn.Linear(100,2)


    def forward(self, x):
        x = self.maxPool(self.conv1(x))
        x = self.maxPool(self.conv2(x))

        x = x.view(x.shape[0],-1)
        x = self.activation(self.Linear1(x))
        x = self.activation(self.Linear2(x))
        x = self.activation(self.Linear3(x))
        x = self.out(x)
        return x