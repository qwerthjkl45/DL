import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.autograd import Variable

from datasets.bmnist import bmnist
from torchvision.utils import save_image

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim);
        self.fc2_mu = torch.nn.Linear(hidden_dim, z_dim);
        self.fc2_std = torch.nn.Linear(hidden_dim, z_dim);
        
        self.relu = nn.ReLU()

    def forward(self, input_data):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        
        #print(self.fc2_std.weight)
        
        h_mu = self.relu(self.fc1(input_data))
        h_std = self.relu(self.fc1(input_data))
        
        mean, std = self.fc2_mu(h_mu), self.fc2_std(h_std)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, output_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(z_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        h = self.relu(self.fc1(input_data))
        mean = self.sigmoid(self.fc2(h))

        return mean


class VAE(nn.Module):

    def __init__(self, input_dim = 28*28, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.input_dim = input_dim;
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(input_dim, hidden_dim, z_dim)
        
        
    def forward(self, input_data):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, std = self.encoder(input_data) # batch_size * 20
        z = self.reparametrize(mean, std) # batch_size * 20
        out = self.decoder(z) # batch_size, 784, which will range between 0 ~1
        
        average_negative_elbo = self.loss_function(out, input_data, mean, std)
        return average_negative_elbo, out

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        z = torch.randn(n_samples, self.z_dim)
        
        sampled_ims = self.decoder(z)
        im_means = torch.mean(sampled_ims, dim=1)

        return sampled_ims, im_means
        
    def reparametrize(self, mu, std):
        epsilon = torch.randn(mu.size()[0], mu.size()[1])    
        return mu + std * epsilon
        
    def loss_function(self, out, input_data, mean, std):
        batch_size = input_data.shape[0]
        average_recon_loss = torch.sum(nn.functional.binary_cross_entropy(out, input_data, reduction = 'none'), dim = 1)
        average_recon_loss = torch.sum(average_recon_loss)/batch_size
        
        
        epsilon =  1e-8
        var = std.pow(2) 
        regular_loss = 0.5 * (-torch.sum(torch.log(epsilon + std**2)) - float(len(mean)) + torch.sum(mean**2) + torch.sum(epsilon + std**2))
        average_negative_elbo = average_recon_loss + (regular_loss/batch_size)
        
        return average_negative_elbo
        
    


def epoch_iter(model, data, optimizer, epoch):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    
    total_loss = 0;
    
    for step, batch_input in enumerate(data):
        batch_input = batch_input.view(-1, 28*28) # batch_size * 784
        optimizer.zero_grad()
        loss, out = model(batch_input)
        loss.backward()      
        optimizer.step()     
        total_loss += loss.item();  
        
    average_epoch_elbo = -total_loss/len(data)
    
    #if model.training:
    #    save_imgs = out.view(-1, 1, 28, 28).cpu()
    #    save_image(save_imgs.data[:25], 'images_vae/%d.png' % epoch, nrow=5, normalize=True)
    
    return average_epoch_elbo


def run_epoch(model, data, optimizer, epoch):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer, epoch)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer, epoch)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    os.makedirs('images_vae', exist_ok=True)
    data = bmnist()[:2]  # ignore test split
            
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer, epoch)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

        samples, samples_mean = model.sample(25)
        save_imgs = samples.view(-1, 1, 28, 28).cpu()
        save_image(save_imgs.data[:25], 'images_vae/%d.png' % epoch, nrow=5, normalize=True)
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    #print(samples_mean)
    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
