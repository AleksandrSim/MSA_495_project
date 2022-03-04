import os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class ImagetoImageDataset(Dataset):
    def __init__(self, domainA_dir, domainB_dir, domainC_dir, domainD_dir, domainE_dir, transforms=None):
        self.imagesA = [os.path.join(domainA_dir, x) for x in os.listdir(domainA_dir) if
                        x.endswith('.png') or x.endswith('jpg')]
        self.imagesB = [os.path.join(domainB_dir, x) for x in os.listdir(domainB_dir) if
                        x.endswith('.png') or x.endswith('jpg')]
        self.imagesC = [os.path.join(domainC_dir, x) for x in os.listdir(domainC_dir) if
                        x.endswith('.png') or x.endswith('jpg')]
        self.imagesD = [os.path.join(domainD_dir, x) for x in os.listdir(domainD_dir) if
                        x.endswith('.png') or x.endswith('jpg')]
        self.imagesE = [os.path.join(domainE_dir, x) for x in os.listdir(domainE_dir) if
                        x.endswith('.png') or x.endswith('jpg')]

        self.transforms = transforms

    def __len__(self):
        return min(len(self.imagesA), len(self.imagesB), len(self.imagesC), len(self.imagesD), len(self.imagesE))

    def __getitem__(self, idx):
        imageA = Image.open(self.imagesA[idx])
        imageB = Image.open(self.imagesB[idx])
        imageC = Image.open(self.imagesC[idx])
        imageD = Image.open(self.imagesD[idx])
        imageE = Image.open(self.imagesE[idx])

        if self.transforms is not None:
            imageA = self.transforms(imageA)
            imageB = self.transforms(imageB)
            imageC = self.transforms(imageC)
            imageD = self.transforms(imageD)
            imageE = self.transforms(imageE)

        return imageA, imageB, imageC, imageD, imageE
