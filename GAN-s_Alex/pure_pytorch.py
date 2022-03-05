import itertools

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from simple_dataloader import ImagetoImageDataset
from GAN_model import Generator, Discriminator
import os
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))



if __name__ == '__main__':
    criterion_GAN = torch.nn.MSELoss()

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
    '''
    d_optim = torch.optim.Adam(itertools.chain(disGA.parameters(),
                                               disGB.parameters()),
                               lr=0.0001,
                               betas=(0.5, 0.999),
                               weight_decay=0.0001)
'''
    optimizer_D_A = torch.optim.Adam(disGA.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(disGB.parameters(), lr=0.0001, betas=(0.5, 0.999))

    df = pd.read_csv(os.path.split(os.getcwd())[0] + '/files/train.txt', sep=' ')
    path_to_images = '/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized/'
    df.columns = ['name', 'age']


    BATCH_SIZE = 3
    dataset = ImagetoImageDataset(df, path_to_images)
    dataset = DataLoader(dataset,
               batch_size=BATCH_SIZE,
               shuffle=True, num_workers=8)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    target_real = Variable(torch.Tensor(BATCH_SIZE).fill_(1.0), requires_grad=False)
    target_fake = Variable(torch.Tensor(BATCH_SIZE).fill_(0.0), requires_grad=False)
    for epoch in range(10):
        for i, batch in enumerate(dataset):
            real_A, real_B= batch
            same_B = genA2B(real_B)
            g_optim.zero_grad()
            loss_identity_B = criterion_identity(same_B, real_B) * 7
            same_A = genB2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 7








            fake_B = genA2B(real_A)



            if i %10 ==0:
                test_image = np.transpose(real_A[0].squeeze().detach().numpy(), [1,2,0])
                plt.imshow(test_image)
                plt.show()


                plt.imshow(test_image)
                aged_face = np.transpose(fake_B[0].squeeze().detach().numpy(), [1,2,0])
                plt.imshow(aged_face)
                plt.show()


                '''
                print('test_image')
                print(test_image.shape)
                read = (real_A.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0

                plt.imshow(read)
                plt.show()

                aged_face = (fake_B.squeeze().permute(1, 2, 0).detach().numpy() + 1.0) / 2.0
                plt.imshow(aged_face)
                plt.show()
'''''

            pred_fake =disGB(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * 2

            fake_A = genB2A(real_B)
            pred_fake = disGA(fake_A)
            loss_GAN_B2A = criterion_identity(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * 2



            recovered_A = genB2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10

            recovered_B = genA2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10



            g_loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            g_loss.backward()

            g_optim.step()




            optimizer_D_A.zero_grad()

            pred_real = disGA(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)


            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = disGA(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)


            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            loss_D_A.backward()


            optimizer_D_A.step()

            optimizer_D_B.zero_grad()

            pred_real = disGB(real_B)


            loss_D_real = criterion_GAN(pred_real, target_real)



            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake =  disGB(fake_B.detach())

            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()




            print('BATCH -------->' + str(i))
            print(' g_loss -------->' + str(g_loss.item()))
            print(' loss_D_B -------->' + str(loss_D_B.item()))
            print(' loss_D_A -------->' + str(loss_D_A.item()))
            print(' loss_cycle -------->' + str(loss_cycle_BAB.item()))












