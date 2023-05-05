import torch
from torch import nn
import torch.nn.functional as F

LATENT = 512

class Truncate(nn.Module):
    def __init__(self, n, beta, threshold=0.8):
        super().__init__()
        self.threshold = threshold
        self.beta = beta

        self.register_buffer('avg_latent', torch.zeros(n))

    def update(self, last_avg):
        self.avg_latent.copy_(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def forward(self, x):
        if self.training:
            self.avg_latent.copy_((1.0 - self.beta) * self.avg_latent + self.beta * x.mean(0).detach())
            return x

        return self.avg_latent.lerp(x, self.threshold)

class EqualizedLinear(nn.Module):
    def __init__(self, a, b, lrmul=1.0, gain=2 ** 0.5):
        super().__init__()
        self.wmul = gain * a ** (-0.5) * lrmul
        self.bmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(b, a) / lrmul)
        self.bias = torch.nn.Parameter(torch.zeros(b))

    def forward(self, x):
        return F.linear(x, self.weight * self.wmul, self.bias * self.bmul)

class EqualizedConv2d(nn.Module):
    def __init__(self, a, b, kn, stride, padding, lrmul=1.0, gain=2 ** 0.5):
        super().__init__()
        self.wmul = gain * (a * kn ** 2) ** (-0.5)
        self.bmul = lrmul

        self.stride = stride
        self.padding = padding

        self.weight = torch.nn.Parameter(torch.randn(b, a, kn, kn) / lrmul)
        self.bias = torch.nn.Parameter(torch.zeros(b))

    def forward(self, x):
        return F.conv2d(x, self.weight * self.wmul, self.bias * self.bmul, self.stride, self.padding)

class NoiseLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.noise = None

    def forward(self, x):
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        x = x + self.weight * noise + self.bias
        return x

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        return x

class Varf(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x.std(0, keepdim=True)
        y = y.mean(1, keepdim=True).expand(x.shape[0], -1).clone()
        z = torch.cat((x, y), dim=1)
        return z

C = 16
class GNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.mapping = nn.Sequential(
            EqualizedLinear(LATENT, 512, lrmul=0.01),

            nn.LeakyReLU(0.2, True),
            EqualizedLinear(512, 512, lrmul=0.01),

            nn.LeakyReLU(0.2, True),
            EqualizedLinear(512, 512, lrmul=0.01),

            nn.LeakyReLU(0.2, True),
            EqualizedLinear(512, 512, lrmul=0.01),

            nn.LeakyReLU(0.2, True),
            EqualizedLinear(512, 512, lrmul=0.01),

            nn.LeakyReLU(0.2, True),
            EqualizedLinear(512, 512, lrmul=0.01),

            Truncate(512, 0.01),
        )

        self.trans = nn.ModuleList([
            nn.Sequential(
                EqualizedLinear(512, C * 16 * 49),
                nn.Unflatten(1, (C * 16, 7, 7)),
            ),

            nn.Sequential(
                EqualizedLinear(512, C * 4 * 14 ** 2),
                nn.Unflatten(1, (C * 4, 14, 14)),
            )
        ])

        self.blocks = nn.ModuleList([
            nn.Sequential(
                NoiseLayer(C * 16),
                EqualizedConv2d(C * 16, C * 16, 3, 1, 1),
                PixelNorm(),
                nn.LeakyReLU(0.2, True),

                NoiseLayer(C * 16),
                EqualizedConv2d(C * 16, C * 16, 3, 1, 1),
                PixelNorm(),
                nn.LeakyReLU(0.2, True),

                nn.PixelShuffle(2),
            ),

            nn.Sequential(
                NoiseLayer(8 * C),
                EqualizedConv2d(8 * C, 4 * C, 3, 1, 1),
                PixelNorm(),
                nn.LeakyReLU(0.2, True),

                NoiseLayer(4 * C),
                EqualizedConv2d(4 * C, 4 * C, 3, 1, 1),
                PixelNorm(),
                nn.LeakyReLU(0.2, True),

                nn.PixelShuffle(2),
            )
        ])

        self.bot = nn.Sequential(
            EqualizedConv2d(C, 1, 1, 1, 0),
            nn.Tanh(),
        )

    def forward(self, z):
        dlatents = self.mapping(z)

        x = None
        for trans, block in zip(self.trans, self.blocks):
            if x is not None:
                x = torch.cat((x, trans(dlatents)), dim=1)
            else:
                x = trans(dlatents)
            x = block(x)

        x = self.bot(x)
        return x

class DNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            EqualizedConv2d(1, C, 3, 1, 1),
            nn.LeakyReLU(0.2, True),

            EqualizedConv2d(C, C, 3, 1, 1),
            nn.LeakyReLU(0.2, True),

            nn.AvgPool2d(2),

            EqualizedConv2d(C, 4 * C, 3, 1, 1),
            nn.LeakyReLU(0.2, True),

            EqualizedConv2d(4 * C, 4 * C, 3, 1, 1),
            nn.LeakyReLU(0.2, True),

            nn.AvgPool2d(2),

            nn.Flatten(),

            EqualizedLinear(C * 4 * 49, 1024),
            nn.LeakyReLU(0.2, True),

            EqualizedLinear(1024, 1024),
            nn.LeakyReLU(0.2, True),

            EqualizedLinear(1024, 1024),
            nn.LeakyReLU(0.2, True),

            EqualizedLinear(1024, 1024),
            Varf(),

            EqualizedLinear(1025, 1024),
            nn.LeakyReLU(0.2, True),

            EqualizedLinear(1024, 1),
        )

    def forward(self, x):
        x = self.net(x)
        return x