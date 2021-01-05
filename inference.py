from model import Generator
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np
import argparse
import torch
import os


if __name__ == '__main__':
    
    os.makedirs("images", exist_ok=True)
    
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
    
    generator = Generator(opt)
    cuda = True if torch.cuda.is_available() else False
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
    
    checkpoint_generator = torch.load('checkpoint/generator.pth.tar', map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint_generator['generator'])
    
    gen_imgs = generator(z)
    save_image(gen_imgs.data[:25], "images/fake2.png", nrow=5, normalize=True)