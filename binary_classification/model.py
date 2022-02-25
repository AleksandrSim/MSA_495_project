import torch.nn as nn
import torch.nn.functional as F



class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,12,5)
        self.conv3 = nn.Conv2d(12,20,5)

        self.maxPool = nn.MaxPool2d(2,2)

        self.Linear1 = nn.Linear(112908, 1000)
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
        self.conv1 = nn.Conv2d(3, 7, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(7, 10 , 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(10*97*97, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 5)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class AgeAlexNet(nn.Module):
    def __init__(self, pretrained=False, modelpath=None):
        super(AgeAlexNet, self).__init__()
        assert pretrained is False or modelpath is not None, "pretrain model need to be specified"
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(2,2e-5,0.75),

            nn.Conv2d(96, 256, kernel_size=5, stride=1,groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(2, 2e-5, 0.75),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3,stride=1,groups=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3,stride=1,groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.age_classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(12544, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 5),
        )
        if pretrained is True:
            self.load_pretrained_params(modelpath)

        self.Conv3_feature_module=nn.Sequential()
        self.Conv4_feature_module=nn.Sequential()
        self.Conv5_feature_module=nn.Sequential()
        self.Pool5_feature_module=nn.Sequential()
        for x in range(10):
            self.Conv3_feature_module.add_module(str(x), self.features[x])
        for x in range(10,12):
            self.Conv4_feature_module.add_module(str(x),self.features[x])
        for x in range(12,14):
            self.Conv5_feature_module.add_module(str(x),self.features[x])
        for x in range(14,15):
            self.Pool5_feature_module.add_module(str(x),self.features[x])


    def forward(self, x):
        self.conv3_feature=self.Conv3_feature_module(x)
        self.conv4_feature=self.Conv4_feature_module(self.conv3_feature)
        self.conv5_feature=self.Conv5_feature_module(self.conv4_feature)
        pool5_feature=self.Pool5_feature_module(self.conv5_feature)
        self.pool5_feature=pool5_feature
        flattened = pool5_feature.view(pool5_feature.size(0), -1)
        age_logit = self.age_classifier(flattened)
        return age_logit