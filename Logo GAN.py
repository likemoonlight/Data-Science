# -*- coding: utf-8 -*-
from __future__ import print_function        
import os
import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import torchvision.transforms as transforms


    # custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
           
            nn.Conv2d(image_color, feature_maps_size_discriminator, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps_size_discriminator, feature_maps_size_discriminator * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size_discriminator * 2),
            nn.LeakyReLU(0.2, inplace=True),
           
            nn.Conv2d(feature_maps_size_discriminator * 2, feature_maps_size_discriminator * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size_discriminator * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps_size_discriminator * 4, feature_maps_size_discriminator * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size_discriminator * 8),
            nn.LeakyReLU(0.2, inplace=True),
           
            nn.Conv2d(feature_maps_size_discriminator * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( input_amount, feature_maps_size_generator  * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps_size_generator  * 8),
            nn.ReLU(True),
            # state size. (feature_maps_size_generator *8) x 4 x 4
            nn.ConvTranspose2d(feature_maps_size_generator  * 8, feature_maps_size_generator  * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size_generator  * 4),
            nn.ReLU(True),
            # state size. (feature_maps_size_generator *4) x 8 x 8
            nn.ConvTranspose2d( feature_maps_size_generator  * 4, feature_maps_size_generator  * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size_generator  * 2),
            nn.ReLU(True),
            # state size. (feature_maps_size_generator *2) x 16 x 16
            nn.ConvTranspose2d( feature_maps_size_generator  * 2, feature_maps_size_generator , 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size_generator ),
            nn.ReLU(True),
            # state size. (feature_maps_size_generator ) x 32 x 32
            nn.ConvTranspose2d( feature_maps_size_generator , image_color, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (image_color) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)



if __name__ == '__main__':


    # Create directory
    output_data_folder ="OutputImages"
#    output_data_folder = sys.argv[2]
    os.mkdir(output_data_folder)

    # Generate random seed
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # dataset dir
#    input_data_folder ="C:/Users/Victor/Desktop/0880304_src/imgs"
    input_data_folder = sys.argv[1]

    # Number of workers for dataloader
    workers = 4

    # Batch size
    batch_size = 256

    # resize
    img_size = 64

    # color images RGB 3
    image_color = 3

    # size of generator input
    input_amount = 100

    # Size of feature maps in generator
    feature_maps_size_generator  = 64

    # Size of feature maps in discriminator
    feature_maps_size_discriminator = 64

    # Number of training epochs
    Num_training  = 200

    # Learning rate for optimizers
    lr = 0.0005

    # Beta1 hyperparam for Adam optimizers
    Beta1 = 0.4

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1
    dtype = torch.cuda.FloatTensor

    dataset = dset.ImageFolder(
        root=input_data_folder,
        transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    # Decide
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)        
    netD.apply(weights_init)


    # Initialize BCELoss function
    loss_function = nn.BCELoss()

    fixed_noise = torch.randn(64, input_amount, 1, 1, device=device ).type(dtype)


    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(Beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(Beta1, 0.999))




    # Training Loop
    iters = 0
    print("Start to Training ...")
    # For each epoch
    for epoch in range(Num_training ):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            

            netD.zero_grad()
            # Forward real
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)

            output = netD(real_cpu).view(-1)

            errD_real = loss_function(output, label)

            errD_real.backward()
            D_x = output.mean().item()

            # ==========================================================================
            # Train fake
            noise = torch.randn(b_size, input_amount, 1, 1, device=device, ).type(dtype)

            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach()).view(-1)

            errD_fake = loss_function(output, label)

            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            
            optimizerD.step()

            # ==========================================================================
            # forward fake
            netG.zero_grad()
            label.fill_(real_label) 
            
            output = netD(fake).view(-1)
            
            errG = loss_function(output, label)
            
            errG.backward()
            D_G_z2 = output.mean().item()
            
            optimizerG.step()

            if i % 50 == 0:
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch+1, Num_training ,
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
                
            iters += 1


    #b_size = real_cpu.size(0)
    i = 0
    while(True):
        noise = torch.randn(64, input_amount, 1, 1, device=device, ).type(dtype)
        # Generate fake image batch with G
        fake = netG(noise).detach().cpu()
        #fake = netG(fixed_noise).detach().cpu()
        tmp = F.interpolate(fake, scale_factor=0.5)
        for j in range(64):
            img = vutils.make_grid(tmp[j,:,:,:], normalize=True)
            i += 1
            save_image(img, './{}/{}.png'.format(output_data_folder,i))
            if(i == 50000):
                break
        if(i == 50000):
            break