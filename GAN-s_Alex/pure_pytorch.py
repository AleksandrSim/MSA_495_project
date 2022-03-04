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



if __name__ == '__main__':





    genA2B = Generator(32, n_residual_blocks=9)
    genB2A = Generator(32, n_residual_blocks=9)
    disGA = Discriminator(32)
    disGB = Discriminator(32)

    # cache for generated images
    generated_A = None
    generated_B = None
    real_A = None
    real_B = None

    g_optim = torch.optim.Adam(itertools.chain(genA2B.parameters(), genB2A.parameters()),
                               lr=0.0001, betas=(0.5, 0.999),
                               weight_decay=0.0001)
    d_optim = torch.optim.Adam(itertools.chain(disGA.parameters(),
                                               disGB.parameters()),
                               lr=0.0001,
                               betas=(0.5, 0.999),
                               weight_decay=0.0001)
    optimizers = [g_optim,d_optim]
    df = pd.read_csv(os.path.split(os.getcwd())[0] + '/files/train.txt', sep=' ')
    path_to_images = '/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized/'
    df.columns = ['name', 'age']

    dataset = ImagetoImageDataset(df, path_to_images)
    dataset = DataLoader(dataset,
               batch_size=1,
               shuffle=True, num_workers=8)


    for epoch in range(10):
        for i, batch in enumerate(dataset):
            real_A, real_B= batch
            for optim_index in range(len(optimizers)):
                if optim_index ==0:
                    same_B = genA2B(real_B)
                    loss_identity_B = F.l1_loss(same_B, real_B) * 7
                    # G_B2A(A) should equal A if real A is fed
                    same_A = genB2A(real_A)
                    loss_identity_A = F.l1_loss(same_A, real_A) * 7

                    fake_B = genA2B(real_A)
                    pred_fake =disGB(fake_B)
                    loss_GAN_A2B = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * 2

                    pred_fake = disGB(fake_B)
                    loss_GAN_A2B = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * 2

                    fake_A = genB2A(real_B)
                    pred_fake = disGA(fake_A)
                    loss_GAN_B2A = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * 2

                    # Cycle loss
                    recovered_A = genB2A(fake_B)
                    loss_cycle_ABA = F.l1_loss(recovered_A, real_A) * 10

                    recovered_B = genA2B(fake_A)
                    loss_cycle_BAB = F.l1_loss(recovered_B, real_B) * 10

                    g_loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB


                    generated_B = fake_B
                    generated_A = fake_A

                    real_B = real_B
                    sreal_A = real_A


                    if  i %3 ==0:
                        print('loss_generator-------->' + str(g_loss))

            if i % 10== 0:
                torch.save(genA2B.state_dict(), '/Users/aleksandrsimonyan/Desktop/models/pure_P_A2B' + str(batch_idx) + '.pth')
                torch.save(genB2A.state_dict(), '/Users/aleksandrsimonyan/Desktop/models/pure_PB2A' + str(batch_idx) + '.pth')


                if optim_index==1:
                    pred_real = disGA(real_A)
                    loss_D_real = F.mse_loss(pred_real, torch.ones(pred_real.shape).type_as(pred_real))

                    # Fake loss
                    fake_A = generated_A
                    pred_fake = disGA(fake_A.detach())
                    loss_D_fake = F.mse_loss(pred_fake, torch.zeros(pred_fake.shape).type_as(pred_fake))

                    # Total loss
                    loss_D_A = (loss_D_real + loss_D_fake) * 0.5

                    # Real loss
                    pred_real = disGB(real_B)
                    loss_D_real = F.mse_loss(pred_real, torch.ones(pred_real.shape).type_as(pred_real))

                    # Fake loss
                    fake_B = generated_B
                    pred_fake = disGB(fake_B.detach())
                    loss_D_fake = F.mse_loss(pred_fake, torch.zeros(pred_fake.shape).type_as(pred_fake))

                    # Total loss
                    loss_D_B = (loss_D_real + loss_D_fake) * 0.5
                    d_loss = loss_D_A + loss_D_B

                    if  i %3 ==0:
                        print('loss_discriminator-------->' + str(d_loss))









