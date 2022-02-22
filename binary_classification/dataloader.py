from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch


class BinaryClass(Dataset):
    def __init__(self, dataset, path_to_images):
        self.dataset = dataset
        self.path_to_images = path_to_images


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        y = torch.tensor(int(self.dataset['old_young'][index])).long()
        X =  Image.open(self.path_to_images+self.dataset['image_name'][index])
        X= self.transform(X)

        return X,y
    transform = transforms.Compose([
        transforms.ToTensor()])


