from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch


class BinaryClass(Dataset):
    def __init__(self, dataset, path_to_images, train = True):
        self.dataset = dataset
        self.path_to_images = path_to_images
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        y = torch.tensor(int(self.dataset['class'][index])).long()

        X = Image.open(self.path_to_images + self.dataset['name'][index])
        if self.train == True:
            X = self.transform["train"](X)
        else:
            X = self.transform["valid"](X)
        return X, y

    transform = {"train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]),
        "valid": transforms.Compose([transforms.ToTensor()])}

if __name__ == '__main__':
    print('hey')

