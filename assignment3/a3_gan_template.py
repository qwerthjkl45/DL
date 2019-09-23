import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable
from torch.distributions import normal
import numpy as np
from scipy.interpolate import interp1d


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        
        #   Linear 1024 -> 768
        #   Output non-linearity
        
        nodes = [args.latent_dim, 128, 256, 512, 1024, 784]
        
        def block(in_feat, out_feat, normalize=True, last_layer = False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            if not last_layer:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
            
        self.fc = []
        for idx in range(len(nodes) - 1):
            if (idx == 0):
                self.fc += [*block(nodes[idx], nodes[idx + 1], normalize= False)]
            elif (idx == len(nodes) - 2):
                self.fc += [*block(nodes[idx], nodes[idx + 1], normalize= False, last_layer= True)]
            else:
                self.fc += [*block(nodes[idx], nodes[idx + 1])]   
        
        self.fc += [nn.Tanh()]        
        self.fc = nn.Sequential(*self.fc)
        

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 28, 28)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        
        nodes = [784, 512, 256, 1]
        self.fc = []
        for idx in range(len(nodes) - 1):
            self.fc += [nn.Linear(nodes[idx], nodes[idx + 1])]
            
            if idx == (len(nodes) - 2):
                break
            self.fc += [nn.LeakyReLU(0.2, inplace=True)]
            
        self.fc += [nn.Sigmoid()]        
        self.fc = nn.Sequential(*self.fc)

    def forward(self, img):
        img = img.view(img.size(0), -1)
        out = self.fc(img)
        
        return out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            
            x = imgs.reshape(imgs.shape[0], 784)
            x.cuda()
            # Adversarial ground truths
            #valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            #fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
            

            # Train Discriminator recognizing fake data
            # ---------------

            optimizer_G.zero_grad()
            #z = normal.Normal(0, 1).sample((imgs.shape[0], args.latent_dim))
            z = torch.randn(imgs.shape[0], args.latent_dim)
            z = z.float()
            gen_imgs = generator(z) # batch_size * 28 * 28
            d_target = torch.ones([imgs.size(0), 1])
            d_loss = nn.functional.binary_cross_entropy(discriminator(gen_imgs), d_target)
            d_loss.backward()
            optimizer_G.step()
            
            
            # Train Discriminator recognizing real data and train generator to fool D
            # -------------------
            
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_target = torch.ones([imgs.size(0), 1])
            fake_target = torch.zeros([imgs.size(0), 1])
            real_loss = nn.functional.binary_cross_entropy(discriminator(x), real_target)
            fake_loss = nn.functional.binary_cross_entropy(discriminator(gen_imgs.detach()), fake_target)
            dg_loss = (real_loss + fake_loss) / 2
            dg_loss.backward()
            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_imgs = gen_imgs.view(-1, 1, 28, 28).cpu()
                save_image(save_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
                
        print(epoch)    


def generator_eval(generator):
    z = torch.zeros([2, args.latent_dim])
    z[0] = normal.Normal(0, 0.1).sample((1, args.latent_dim))
    z[1] = normal.Normal(0, 1).sample((1, args.latent_dim))
    z = z.float()
    
    z_diff = (z[0] - z[1])/9.0
    z_img = torch.zeros([9, args.latent_dim])
    
    # 7 interpolation steps
    for idx in range(9):
        z_img[idx] = z[0] + (idx*z_diff)
        
    gen_imgs = generator(z_img) # batch_size * 28 * 28
    save_imgs = gen_imgs.view(-1, 1, 28, 28).cpu()
    save_image(save_imgs.data[0:9], 'images/1.png' , nrow = 9 , normalize=True)
    

def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    if not (args.train):
        generator.load_state_dict(torch.load('./mnist_generator.pt'));
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    if (args.train):
        train(dataloader, discriminator, generator, optimizer_G, optimizer_D)
    else:
        print('eval')
        generator.eval()
        generator_eval(generator)    
    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--train', type=bool, default=True,
                        help='train or eval')
    args = parser.parse_args()

    main()
