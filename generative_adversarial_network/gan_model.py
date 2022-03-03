import torch
import torch.nn as nn
import itertools
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.utils import make_grid
from gan_data import *


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.BatchNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, ngf, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, ngf, 7),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU()]

        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU()]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU()]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, 3, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(3, ndf, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
                  nn.BatchNorm2d(ndf * 2),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(ndf * 4),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf * 4, ndf * 8, 4, padding=1),
                  nn.InstanceNorm2d(ndf * 8),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(ndf * 8, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class AgingGAN(pl.LightningModule):

    def __init__(self, hparams):
        super(AgingGAN, self).__init__()
        self.hparams = hparams
        self.genA2B = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        self.genB2A = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        self.disGA = Discriminator(hparams['ndf'])
        self.disGB = Discriminator(hparams['ndf'])

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
            loss_identity_B = F.l1_loss(same_B, real_B) * self.hparams['identity_weight']
            # G_B2A(A) should equal A if real A is fed
            same_A = self.genB2A(real_A)
            loss_identity_A = F.l1_loss(same_A, real_A) * self.hparams['identity_weight']

            # GAN loss
            fake_B = self.genA2B(real_A)
            pred_fake = self.disGB(fake_B)
            loss_GAN_A2B = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * self.hparams[
                'adv_weight']

            fake_A = self.genB2A(real_B)
            pred_fake = self.disGA(fake_A)
            loss_GAN_B2A = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * self.hparams[
                'adv_weight']

            # Cycle loss
            recovered_A = self.genB2A(fake_B)
            loss_cycle_ABA = F.l1_loss(recovered_A, real_A) * self.hparams['cycle_weight']

            recovered_B = self.genA2B(fake_A)
            loss_cycle_BAB = F.l1_loss(recovered_B, real_B) * self.hparams['cycle_weight']

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
                self.logger.experiment.add_image('Generated/A',
                                                 make_grid(self.generated_A, normalize=True, scale_each=True),
                                                 self.current_epoch)
                self.logger.experiment.add_image('Generated/B',
                                                 make_grid(self.generated_B, normalize=True, scale_each=True),
                                                 self.current_epoch)
            return output

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
                                   lr=self.hparams['lr'], betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        d_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(),
                                                   self.disGB.parameters()),
                                   lr=self.hparams['lr'],
                                   betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        return [g_optim, d_optim], []

    def train_dataloader(self):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.hparams['img_size'] + 30, self.hparams['img_size'] + 30)),
            transforms.RandomCrop(self.hparams['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        dataset = ImagetoImageDataset(self.hparams['domainA_dir'], self.hparams['domainB_dir'], train_transform)
        return DataLoader(dataset,
                          batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'],
                          shuffle=True)
