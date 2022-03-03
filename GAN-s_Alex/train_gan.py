from pytorch_lightning import Trainer
from gan_module import AgingGAN
model = AgingGAN()
trainer = Trainer(max_epochs=10, auto_scale_batch_size='binsearch')