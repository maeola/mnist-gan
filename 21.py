import os
import coloredlogs, logging

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from nets import *

import torchvision

import matplotlib.pyplot as plt

coloredlogs.install(fmt='%(levelname)7s %(asctime)s %(hostname)s %(message)s')

device = torch.device('cuda')

logging.info(f'Using device {device}')

BATCH_SIZE = 256
GAMMA = 0.001
LR = 1e-3
LOG_INT = 500

DATA = os.environ['DATA']

train = torchvision.datasets.MNIST(f'{DATA}/vision', train=True, download=True)
train = train.data / 255.0
train = train * 2 - 1
train = train.unsqueeze(1)

logging.info(train.shape)

train = train.to(torch.float).to(device)

G = GNet().to(device)
D = DNet().to(device)

if 0:
    data = torch.load('bruh.pt')
    G.load_state_dict(data['G'])
    D.load_state_dict(data['D'])
    logging.warning('Loaded G and D from file ...')

GO = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.0, 0.9))
DO = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.0, 0.9))

def genlatent(n):
    z = torch.randn(n, LATENT, dtype=torch.float, device=device)
    z = z * (z.square().mean(1, keepdim=True) + 1e-8).rsqrt()
    return z

def plot_latent_space(epoch, imgs, n=30, figsize=15):
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    for i in range(n):
        for j in range(n):
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = imgs[i * n + j]

    fig = plt.figure(figsize=(figsize, figsize), frameon=False)
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(f'output/{epoch}.png', dpi=150)
    plt.close(fig)

wd_loss = 0
wg_loss = 0
count = 0

platent = genlatent(900)
while True:
    count += 1
    gamma = max(1 / count, GAMMA)

    G.train()
    D.train()

    idx = torch.randint(train.shape[0], (BATCH_SIZE,))
    real = train[idx]
    fake = G(genlatent(BATCH_SIZE))

    DO.zero_grad()

    d_real = D(real)
    d_fake = D(fake.detach())
    d_loss = (d_fake - d_real + 0.25 * d_real ** 2 + 0.25 * d_fake ** 2).mean()
    d_loss.backward()

    wd_loss = d_loss.item() * gamma + wd_loss * (1 - gamma)

    DO.step()

    GO.zero_grad()

    g_loss = -D(fake).mean()
    g_loss.backward()
    wg_loss = g_loss.item() * gamma + wg_loss * (1 - gamma)

    GO.step()

    if count % LOG_INT == 0:
        G.eval()
        with torch.no_grad():
            imgs = G(platent).cpu().reshape(-1, 28, 28)
            imgs = 0.5 * imgs + 0.5
        # imgs = train[torch.randint(train.shape[0], (900,))].cpu()
        plot_latent_space(epoch=count, imgs=imgs)

        logging.info(f'{count} {wd_loss:.3f} {wg_loss:.3f}')
        torch.save({'G': G.state_dict(), 'D': D.state_dict()}, 'bruh.pt')
