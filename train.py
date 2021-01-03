from model import Generator, Discriminator
from tensorboardX import SummaryWriter
from data_loader import get_data_loader

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn.functional as F
import torch.nn as nn
from utils import *
import numpy as np

import argparse
import os
import torch


if __name__ == "__main__":

    os.makedirs("images", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("tensorboard", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels, 3 for RGB image")
    parser.add_argument("--sample_interval", type=int, default=1000, help="number of image channels")
    parser.add_argument("--tensorboard", type=str, default="tensorboard/losses", help="where losses are located")
    parser.add_argument("--resume_generator", type=str, default=None, help="resume generator")
    parser.add_argument("--resume_discriminator", type=str, default=None, help="discriminator")
    opt = parser.parse_args()
    print(opt)

    writer = SummaryWriter(opt.tensorboard)
    cuda = True if torch.cuda.is_available() else False
    
    # !!! Minimizes MSE instead of BCE
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    dataloader = get_data_loader(opt)

    # optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # resume checkpoint
    if opt.resume_generator and opt.resume_discriminator:
        print('Resuming checkpoint from {} and {}'.format(opt.resume_generator, opt.resume_discriminator))
        checkpoint_generator = torch.load(opt.resume_generator)
        checkpoint_discriminator = torch.load(opt.resume_discriminator)

        generator.load_state_dict(checkpoint_generator['generator'])
        discriminator.load_state_dict(checkpoint_discriminator['discriminator'])

        print('Validating the checkpoints ... ')

    batch_idx = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_idx += 1

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -------------------------------------Train Generator------------------------------------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            writer.add_scalar("g_loss: ", g_loss.cpu(), batch_idx)

            g_loss.backward()
            optimizer_G.step()

            # -------------------------------------Train Discriminator---------------------------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            writer.add_scalar("real_loss: ", real_loss.cpu(), batch_idx)
            writer.add_scalar("fake_loss: ", fake_loss.cpu(), batch_idx)
            writer.add_scalar("d_loss: ", d_loss.cpu(), batch_idx)

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

        if epoch % 20 == 0:
            save_checkpoint({
                'epoch': epoch,
                'generator': generator.state_dict()
            }, 'checkpoint/generator{}.pth.tar'.format(epoch))
        
            save_checkpoint({
                'epoch': epoch,
                'discriminator': discriminator.state_dict()
            }, 'checkpoint/discriminator{}.pth.tar'.format(epoch))
