

from pytorch_lightning import Trainer
from gan_module import AgingGAN
import cv2
model = AgingGAN()
import os

if __name__ == "__main__":
    print(os.listdir('/home/alex/cross_age_dataset_cleaned_and_resized/'))


    trainer = Trainer(max_epochs=10, gpus=0)
    trainer.fit(model)
