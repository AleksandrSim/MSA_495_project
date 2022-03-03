import torchvision


from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms


class ImagetoImageDataset(Dataset):
    def __init__(self, df, path_to_images):
        self.image_A = df[(df['age'] >= 18) & (df['age'] <=30 )].reset_index(drop = True)
        self.image_B = df[(df['age'] >=55)].reset_index(drop = True)
        self.path_to_images = path_to_images

        self.transforms =  torchvision.transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((400 + 30, 400 + 30)),
            transforms.RandomCrop(400),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return min(len(self.imagesA), len(self.imagesB))


    def __getitem__(self, index):
        imageA = Image.open(self.path_to_images + self.image_A['name'][index])
        imageB = Image.open(self.path_to_images + self.image_B['name'][index])
        imageA = self.transforms(imageA)
        imageB = self.transforms(imageB)

        return imageA, imageB