import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

z_dim = 32        # Latent Dimensionality
input_channels = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, z_dim, channels, generator_features=32):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, generator_features * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(generator_features * 4, generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( generator_features * 2, generator_features * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d( generator_features * 1, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self, channels, discriminator_features=32):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_features, discriminator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_features * 2, discriminator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_features * 4, 1, 4, 1, 0, bias=False),
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.model(input)

generator = Generator(z_dim, input_channels).to(device)
discriminator = Discriminator(input_channels).to(device)

gen_lr = 1e-4     # Learning Rate for the Generator
disc_lr = 1e-4    # Learning Rate for the Discriminator

discriminator_optimizer = Adam(discriminator.parameters(), lr=disc_lr, betas=(0.5, 0.999))
generator_optimizer = Adam(generator.parameters(), lr=gen_lr, betas=(0.5, 0.999))

criterion = nn.BCEWithLogitsLoss()

def discriminator_train(discriminator, generator, real_samples, fake_samples):
  # Takes as input real and fake samples and returns the loss for the discriminator
  # Inputs:
  #   real_samples: Input images of size (batch_size, 3, 32, 32)
  #   fake_samples: Input images of size (batch_size, 3, 32, 32)
  # Returns:
  #   loss: Discriminator loss

  ones = torch.ones(real_samples.size(0)).to(device)   # WRITE CODE HERE (targets for real data)
  zeros = torch.zeros(fake_samples.size(0)).to(device) # WRITE CODE HERE (targets for fake data)

  real_output = discriminator(real_samples).view(-1)   # WRITE CODE HERE (output of discriminator on real data)
  fake_output = discriminator(fake_samples).view(-1)   # WRITE CODE HERE (output of discriminator on fake data)

  loss = criterion(real_output, ones) + criterion(fake_output, zeros)
  
  return loss

def generator_train(discriminator, generator, fake_samples):
  # Takes as input fake samples and returns the loss for the generator
  # Inputs:
  #   fake_samples: Input images of size (batch_size, 3, 32, 32)
  # Returns:
  #   loss: Generator loss

  ones = torch.ones(fake_samples.size(0)).to(device)   # WRITE CODE HERE (targets for fake data but for generator loop)

  output = discriminator(fake_samples).view(-1)

  loss = criterion(output, ones) # WRITE CODE HERE (loss for the generator based on criterion and above variables)

  return loss

def sample(generator, num_samples, noise=None):
  # Takes as input the number of samples and returns that many generated samples
  # Inputs:
  #   num_samples: Scalar denoting the number of samples
  # Returns:
  #   samples: Samples generated; tensor of shape (num_samples, 3, 32, 32)

  with torch.no_grad():
    # WRITE CODE HERE (sample from p_z and then generate samples from it)
    if noise is None:
      noise = torch.randn(num_samples, z_dim, 1, 1).to(device)
    samples = generator(noise)
  return samples


def interpolate(generator, z_1, z_2, n_samples):
  # Interpolate between z_1 and z_2 with n_samples number of points, with the first point being z_1 and last being z_2.
  # Inputs:
  #   z_1: The first point in the latent space
  #   z_2: The second point in the latent space
  #   n_samples: Number of points interpolated
  # Returns:
  #   sample: A sample from the generator obtained from each point in the latent space
  #           Should be of size (n_samples, 3, 32, 32)

  # WRITE CODE HERE (interpolate z_1 to z_2 with n_samples points and then)
  with torch.no_grad():
    z = torch.zeros(n_samples, z_dim, 1, 1).to(device)
    for i in range(n_samples):
      z[i] = z_1 + (z_2 - z_1) * i / (n_samples - 1)
    sample = generator(z)
  # WRITE CODE HERE (    generate samples from the respective latents     )
  
  return sample
