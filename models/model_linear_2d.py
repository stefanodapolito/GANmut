import torch
import torch.nn as nn
import numpy as np
import sys


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

    def __init__(self, device, conv_dim=64, c_dim=8, repeat_num=6, n_r=5):
        super(Generator, self).__init__()

        self.nr = n_r
        self.c_dim = c_dim
        self.device = device
        # the six axes, real weight are 6X2
        self.axes = nn.Linear(2, c_dim - 1)
        # make the weight small so that they can easily modified by gradient descend
        self.axes.weight.data = self.axes.weight.data * 0.0001

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

    def forward(
        self,
        x,
        c,
        expr_strength,
        mode="train",
        manual_expr=None,
    ):

        """
        mode can be:

            1) random: code is completely random
            2) manual_selection: code is given manually
            3) train: first nr direction ar choosen randomly
            4) test: no direction is choosen randomly
        """

        if mode == "random":

            n_random = x.size(0)
            angle = torch.rand(n_random, device=self.device) * (2 * np.pi)

            expr_strength = torch.rand(n_random, device=self.device)

            random_vector = torch.empty((n_random, 2), device=self.device)

            random_vector[:, 0] = torch.cos(angle) * expr_strength[:n_random]
            random_vector[:, 1] = torch.sin(angle) * expr_strength[:n_random]

            expr2 = random_vector.view(c.size(0), 2, 1, 1)
            expr3 = expr2.repeat(1, 1, x.size(2), x.size(3))

            x = torch.cat([x, expr3], dim=1)
            return self.main(x), random_vector

        else:

            axes_normalized = nn.functional.normalize(self.axes.weight, p=2, dim=1)

            # axis selection
            if not mode == "manual_selection":
                axis = torch.mm(
                    c[:, 1 : self.c_dim], axes_normalized
                )  # axis 0 is neutral and so must be set to 0

            if mode == "train":
                expr = (axis.transpose(0, 1) * expr_strength).transpose(
                    0, 1
                ) + torch.randn(c.size(0), 2, device=self.device) * 0.075
                if x.size(0) >= self.nr:
                    n_random = min(self.nr, x.size(0))
                    angle = torch.rand(n_random, device=self.device) * (2 * np.pi)
                    random_vector = torch.empty((n_random, 2), device=self.device)

                    random_vector[:, 0] = torch.cos(angle) * expr_strength[:n_random]
                    random_vector[:, 1] = torch.sin(angle) * expr_strength[:n_random]

                    expr[:n_random, :] = random_vector

            elif mode == "manual_selection":
                expr = manual_expr

            elif mode == "test":
                expr = (axis.transpose(0, 1) * expr_strength).transpose(0, 1)

            else:

                sys.exit(
                    "Modality can be only 'random','manual_selection','train','test'."
                )

            expr2 = expr.view(x.size(0), 2, 1, 1)  # put c.size(0) if bug!!!!!!!
            expr3 = expr2.repeat(1, 1, x.size(2), x.size(3))

            x = torch.cat([x, expr3], dim=1)
            return self.main(x), expr

    def print_axes(self):

        print("AXES")
        print(nn.functional.normalize(self.axes.weight, p=2, dim=1))


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
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        self.conv3 = nn.Conv2d(curr_dim, 2, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        out_expr_strength = self.conv3(h)
        return (
            out_src,
            out_cls.view(out_cls.size(0), out_cls.size(1)),
            out_expr_strength.view(out_expr_strength.size(0), 2),
        )
