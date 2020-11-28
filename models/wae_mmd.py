from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseVAE


class WAE_MMD(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        reg_weight: int = 100,
        kernel_type: str = "imq",
        latent_var: float = 2.0,
        lr: float = 0.005,
        weight_decay: Optional[float] = 0,
        scheduler_gamma: Optional[float] = 0.95,
    ) -> None:
        super(WAE_MMD, self).__init__(
            lr=lr, weight_decay=weight_decay, scheduler_gamma=scheduler_gamma
        )

        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.Tensor) Input tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z = self.fc_z(result)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        z = self.encode(input)
        return [self.decode(z), input, z]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr

        recons_loss = F.mse_loss(recons, input)

        mmd_loss = self.compute_mmd(z, reg_weight)

        loss = recons_loss + mmd_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "MMD": mmd_loss}

    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == "rbf":
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == "imq":
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError("Undefined kernel type.")

        return result

    def compute_rbf(
        self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (torch.Tensor)
        :param x2: (torch.Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2.0 * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(
        self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (torch.Tensor)
        :param x2: (torch.Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z: torch.Tensor, reg_weight: float) -> torch.Tensor:
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = (
            reg_weight * prior_z__kernel.mean()
            + reg_weight * z__kernel.mean()
            - 2 * reg_weight * priorz_z__kernel.mean()
        )
        return mmd

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
