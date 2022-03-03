from pytorch_lightning import Trainer
from gan_module import AgingGAN
model = AgingGAN()


if __name__ == "__main__":
    trainer = Trainer(max_epochs=10, gpus=0)
    trainer.fit(model)
