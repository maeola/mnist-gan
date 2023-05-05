import os
import logging

import numpy as np

import torch
from torch import nn

import torchvision

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

device = torch.device('cuda')

logging.info(f'Using device {device}')

BATCH_SIZE = 256
GAMMA = 0.001
LR = 1e-4
LATENT = 64
LOG_INT = 50

DATA = os.environ['DATA']

train = torchvision.datasets.MNIST(f'{DATA}/vision', train=True, download=True)
train = train.data / 255.0
train = train * 2 - 1
train = train.unsqueeze(1)

logging.info(train.shape)


C = 64
class GNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(LATENT, 2 * C * 49),

            nn.Unflatten(1, (2 * C, 7, 7)),

            nn.ConvTranspose2d(2 * C, C, 4, 2, 1),
            nn.ReLU(True),

            nn.ConvTranspose2d(C, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class DNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, C, 4, 2, 1),
            nn.ReLU(True),

            nn.Conv2d(C, 2 * C, 4, 2, 1),
            nn.ReLU(True),

            nn.Flatten(),

            nn.Linear(2 * C * 49, 1),
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

wd_loss = 100
wg_loss = 100
count = 0

lossf = nn.BCELoss()

while True:
    count += 1
    gamma = max(1 / count, GAMMA)

    G.train()
    D.train()

    idx = torch.randint(train.shape[0], (BATCH_SIZE,))
    real = train[idx]
    fake = gen(BATCH_SIZE)

    vf = torch.full((BATCH_SIZE, 1,), 1, dtype=torch.float, device=device)
    ff = torch.full((BATCH_SIZE, 1,), 0, dtype=torch.float, device=device)

    DO.zero_grad()

    d_loss = (lossf(D(real), vf) + lossf(D(fake.detach()), ff)) * 0.5
    d_loss.backward()

    DO.step()
    wd_loss = d_loss.item() * gamma + wd_loss * (1 - gamma)

    GO.zero_grad()

    g_loss = lossf(D(fake), vf)
    g_loss.backward()

    GO.step()
    wg_loss = g_loss.item() * gamma + wg_loss * (1 - gamma)

    if count % LOG_INT == 0:
        logging.info(f'{count} {wd_loss:.3f} {wg_loss:.3f}')

    if count > 15000:
        G.eval()
        plot_latent_space()
        break

