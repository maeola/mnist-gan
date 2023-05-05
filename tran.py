import os
import coloredlogs, logging

import numpy as np

import torch
from torch import nn

import torchvision

import matplotlib.pyplot as plt

coloredlogs.install(fmt='%(levelname)7s %(asctime)s %(hostname)s %(message)s')

device = torch.device('cuda')

logging.info(f'Using device {device}')

BATCH_SIZE = 256
GAMMA = 0.001
LR = 1e-4
LATENT = 256
LOG_INT = 50
eps = 1e-6

DATA = os.environ['DATA']

train = torchvision.datasets.MNIST(f'{DATA}/vision', train=True, download=True)
train = train.data / 255.0
train = train * 2 - 1
train = train.unsqueeze(1)

logging.info(train.shape)

class Attention(nn.Module):
    def __init__(self, n) -> None:
        super().__init__()

        self.vn = nn.Linear(n, n)
        self.kn = nn.Linear(n, n)
        self.qn = nn.Linear(n, n)

        self.net = nn.Sequential(
            nn.Linear(n, 4 * n),
            nn.LeakyReLU(0.1, True),

            nn.Linear(4 * n, n),
        )

        self.norm = nn.LayerNorm(n)

    def forward(self, x):
        x = self.norm(x)
        v = self.vn(x)
        k = self.kn(x)
        q = self.qn(x)

        w = (k @ q.T) / np.sqrt(k.shape[-1])
        w = torch.softmax(w, dim=-1)

        h = w @ v

        return self.net(h) + x

class GNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(LATENT, 16 * 49),

            nn.Unflatten(1, (16, 7, 7)),

            nn.PixelShuffle(2),
            nn.LayerNorm((14, 14)),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(4, 4, 3, 1, 1),
            nn.LeakyReLU(0.1, True),

            nn.PixelShuffle(2),
            nn.LayerNorm((28, 28)),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(1, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.net(x)
        return x



class DNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.1, True),

            nn.LayerNorm((14, 14)),
            nn.Conv2d(4, 16, 3, 1, 1),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.1, True),

            nn.Flatten(),

            nn.Linear(16 * 49, 512),
            nn.LeakyReLU(0.1, True),

            Attention(512),

            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


train = train.to(torch.float).to(device)

G = GNet().to(device)
D = DNet().to(device)
GO = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
DO = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

def gen(n):
    z = torch.randn(n, LATENT, dtype=torch.float, device=device)
    return G(z)

def plot_latent_space(n=30, figsize=15):
    digit_size = 28
    scale = 1
    figure = np.zeros((digit_size * n, digit_size * n))

    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            with torch.no_grad():
                x_decoded = gen(1)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            digit = torch.clamp(digit * 0.5 + 0.5, 0.0, 1.0)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit.cpu()

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

wd_loss = 0
wg_loss = 0
count = 0

while True:
    count += 1
    gamma = max(1 / count, GAMMA)

    G.train()
    D.train()

    idx = torch.randint(train.shape[0], (BATCH_SIZE,))
    real = train[idx]
    fake = gen(BATCH_SIZE)

    DO.zero_grad()        

    d_real = D(real)
    d_fake = D(fake.detach())
    d_loss = -(torch.log(1 - d_fake + eps) + torch.log(d_real + eps)).mean()
    d_loss.backward()
    wd_loss = d_loss.item() * gamma + wd_loss * (1 - gamma)

    DO.step()

    GO.zero_grad()

    g_loss = -torch.log(D(fake) + eps).mean()
    g_loss.backward()
    wg_loss = g_loss.item() * gamma + wg_loss * (1 - gamma)

    GO.step()

    if count % LOG_INT == 0:
        logging.info(f'{count} {wd_loss:.3f} {wg_loss:.3f}')

    if count > 3000:
        G.eval()
        plot_latent_space()
        break
