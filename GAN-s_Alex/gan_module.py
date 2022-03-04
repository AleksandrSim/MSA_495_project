import itertools

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from simple_dataloader import ImagetoImageDataset
from GAN_model import Generator, Discriminator
import os
import pandas as pd
import matplotlib.pyplot as plt

path_to_images = '/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized/'
class AgingGAN(pl.LightningModule):

    def __init__(self):
        super(AgingGAN, self).__init__()
        self.genA2B = Generator(32, n_residual_blocks=9)
        self.genB2A = Generator(32, n_residual_blocks=9)
        self.disGA = Discriminator(32)
        self.disGB = Discriminator(32)

        # cache for generated images
        self.generated_A = None
        self.generated_B = None
        self.real_A = None
        self.real_B = None

    def forward(self, x):
        return self.genA2B(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B = batch

        if optimizer_idx == 0:
            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = self.genA2B(real_B)
            loss_identity_B = F.l1_loss(same_B, real_B) * 7
            # G_B2A(A) should equal A if real A is fed
            same_A = self.genB2A(real_A)
            loss_identity_A = F.l1_loss(same_A, real_A) * 7

            # GAN loss
            fake_B = self.genA2B(real_A)
            if batch_idx %30 ==0:
                read = (real_A.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
                plt.imshow(read)
                plt.show()




                aged_face = (fake_B.squeeze().permute(1, 2, 0).detach().numpy() + 1.0) / 2.0
                plt.imshow(aged_face)
                plt.show()

            pred_fake = self.disGB(fake_B)
            loss_GAN_A2B = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * 2

            fake_A = self.genB2A(real_B)
            pred_fake = self.disGA(fake_A)
            loss_GAN_B2A = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * 2

            # Cycle loss
            recovered_A = self.genB2A(fake_B)
            loss_cycle_ABA = F.l1_loss(recovered_A, real_A) * 10

            recovered_B = self.genA2B(fake_A)
            loss_cycle_BAB = F.l1_loss(recovered_B, real_B) * 10

            # Total loss
            g_loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            output = {
                'loss': g_loss,
                'log': {'Loss/Generator': g_loss}
            }
            self.generated_B = fake_B
            self.generated_A = fake_A

            self.real_B = real_B
            self.real_A = real_A

            # Log to tb
            if batch_idx % 500 == 0:
                self.logger.experiment.add_image('Real/A', make_grid(self.real_A, normalize=True, scale_each=True),
                                                 self.current_epoch)
                self.logger.experiment.add_image('Real/B', make_grid(self.real_B, normalize=True, scale_each=True),
                                                 self.current_epoch)
              #  self.logger.experiment.add_image('Generated/A',
             #                                    make_grid(self.generated_A, normalize=True, scale_each=True),
            #                                     self.current_epoch)
         #       self.logger.experiment.add_image('Generated/B',
          #                                       make_grid(self.generated_B, normalize=True, scale_each=True),
           #                                      self.current_epoch)
            return output
        if batch_idx % 10== 0:
            torch.save(self.genA2B.state_dict(), '/Users/aleksandrsimonyan/Desktop/models/A2B' + str(batch_idx) + '.pth')
            torch.save(self.genA2B.state_dict(), '/Users/aleksandrsimonyan/Desktop/models/B2A' + str(batch_idx) + '.pth')


        if optimizer_idx == 1:
            # Real loss
            pred_real = self.disGA(real_A)
            loss_D_real = F.mse_loss(pred_real, torch.ones(pred_real.shape).type_as(pred_real))

            # Fake loss
            fake_A = self.generated_A
            pred_fake = self.disGA(fake_A.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros(pred_fake.shape).type_as(pred_fake))

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            # Real loss
            pred_real = self.disGB(real_B)
            loss_D_real = F.mse_loss(pred_real, torch.ones(pred_real.shape).type_as(pred_real))

            # Fake loss
            fake_B = self.generated_B
            pred_fake = self.disGB(fake_B.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros(pred_fake.shape).type_as(pred_fake))

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            d_loss = loss_D_A + loss_D_B
            output = {
                'loss': d_loss,
                'log': {'Loss/Discriminator': d_loss}
            }
            return output

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                   lr=0.0001, betas=(0.5, 0.999),
                                   weight_decay=0.0001)
        d_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(),
                                                   self.disGB.parameters()),
                                   lr=0.0001,
                                   betas=(0.5, 0.999),
                                   weight_decay=0.0001)
        return [g_optim, d_optim], []

    def train_dataloader(self):
        df = pd.read_csv(os.path.split(os.getcwd())[0] + '/files/train.txt', sep=' ')
        df.columns = ['name', 'age']
        print(df)
        dataset = ImagetoImageDataset(df, path_to_images)
        return DataLoader(dataset,
                          batch_size=1,
                          shuffle=True, num_workers = 8)