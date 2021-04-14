import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, device, conv_dim=64, c_dim=7, repeat_num=6):

        super(Generator, self).__init__()

        self.device = device
        self.c_dim = c_dim
        # the six axes, real weight are 6X2
        
        self.covariances_list=[]
        for emotion in range(c_dim):
            self.covariances_list.append(nn.Linear(3,3), bias=False)
            
        self.mu = nn.Linear(3, c_dim, bias=False)
        self.covariance_axes.weight.data.fill_(1.0)
            
        layers = []
        layers.append(
            nn.Conv2d(3 + 2, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        )
        layers.append(
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        )
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True)
            )
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(
                nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True)
            )
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        )
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, expr=None, label_trg=None):

        # mo=torch.Tensor([-1.,-1.],device=self.device)
        # po=torch.Tensor([1.,1.],device=self.device)

        batch_size = x.size(0)

        # case when one watn to reproduce a basic emotion
        if label_trg is not None:

            expr = torch.empty((batch_size, 3), device=self.device)

            for batch_sample in range(batch_size):

                expr[batch_sample, :] = self.mu.weight[label_trg[batch_sample]]

        if expr is None:

            expr = torch.empty((batch_size, 3), device=self.device)
            batch_size = x.size(0)
            if batch_size >= self.c_dim:
                expr[: self.c_dim, :] = torch.tanh(self.mu.weight)
                expr[self.c_dim :] = (
                    torch.rand((batch_size - self.c_dim, 3), device=self.device) * 2 - 1
                )

            else:
                expr[:batch_size, :] = torch.tanh(self.mu.weight[:batch_size])

        

        
        C_inv_list = [torch.inverse(C.weight) for C in self.covariances_list]

        # vector of unnrmalized distances
        rep_mu = torch.tanh(self.mu.weight.unsqueeze(0).repeat(x.size(0), 1, 1))
        rep_expr = expr.unsqueeze(1).repeat(1, self.c_dim, 1)
        vector = rep_expr - rep_mu

        mahalanobis_distances = torch.empty(batch_size, self.c_dim, device=self.device)

        
        # mahalanobis_distance^2=vec*C_inv*vec
        for el in range(batch_size):
            for ex in range(self.c_dim):
                mahalanobis_distances[el, ex] = (
                    torch.matmul(vector[el, ex], torch.matmul(C_inv_list[ex],vector[el, ex])))

        # reshaping expr
        expr2 = expr.view(x.size(0), 3, 1, 1)
        expr3 = expr2.repeat(1, 1, x.size(2), x.size(3))
        # produce imagines
        x = torch.cat([x, expr3], dim=1)
        return self.main(x), mahalanobis_distances, expr

    def print_expr(self):

        print("MU")
        print(torch.tanh(self.mu.weight))
        print("Covariance Matrix angles")
        print(self.covariance_angles.weight)
        print("covariance matrix axes")
        print(self.covariance_axes.weight)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(
            curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(curr_dim, c_dim + 2, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
