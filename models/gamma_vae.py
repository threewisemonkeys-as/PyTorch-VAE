from typing import List, Optional

import torch
from torch import nn
from torch.distributions import Gamma
from torch.nn import functional as F

from .base import BaseVAE


class GammaVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        gamma_shape: float = 8.0,
        prior_shape: float = 2.0,
        prior_rate: float = 1.0,
        lr: float = 0.005,
        weight_decay: Optional[float] = 0,
        scheduler_gamma: Optional[float] = 0.95,
    ) -> None:
        super(GammaVAE, self).__init__(
            lr=lr, weight_decay=weight_decay, scheduler_gamma=scheduler_gamma
        )
        self.latent_dim = latent_dim
        self.B = gamma_shape

        self.prior_alpha = torch.tensor([prior_shape])
        self.prior_beta = torch.tensor([prior_rate])

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
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 4, latent_dim), nn.Softmax()
        )
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 4, latent_dim), nn.Softmax()
        )

        # Build Decoder
        modules = []

        self.decoder_input = nn.Sequential(nn.Linear(latent_dim, hidden_dims[-1] * 4))

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
            nn.Sigmoid(),
        )

        self.weight_init()

    def weight_init(self):

        # print(self._modules)
        for block in self._modules:
            for m in self._modules[block]:
                init_(m)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
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
        alpha = self.fc_mu(result)
        beta = self.fc_var(result)

        return [alpha, beta]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Reparameterize the Gamma distribution by the shape augmentation trick.
        Reference:
        [1] https://arxiv.org/pdf/1610.05683.pdf

        :param alpha: (torch.Tensor) Shape parameter of the latent Gamma
        :param beta: (torch.Tensor) Rate parameter of the latent Gamma
        :return:
        """
        # Sample from Gamma to guarantee acceptance
        alpha_ = alpha.clone().detach()
        z_hat = Gamma(alpha_ + self.B, torch.ones_like(alpha_)).sample()

        # Compute the eps ~ N(0,1) that produces z_hat
        eps = self.inv_h_func(alpha + self.B, z_hat)
        z = self.h_func(alpha + self.B, eps)

        # When beta != 1, scale by beta
        return z / beta

    def h_func(self, alpha: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Reparameterize a sample eps ~ N(0, 1) so that h(z) ~ Gamma(alpha, 1)
        :param alpha: (torch.Tensor) Shape parameter
        :param eps: (torch.Tensor) Random sample to reparameterize
        :return: (torch.Tensor)
        """

        z = (alpha - 1.0 / 3.0) * (1 + eps / torch.sqrt(9.0 * alpha - 3.0)) ** 3
        return z

    def inv_h_func(self, alpha: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse reparameterize the given z into eps.
        :param alpha: (torch.Tensor)
        :param z: (torch.Tensor)
        :return: (torch.Tensor)
        """
        eps = torch.sqrt(9.0 * alpha - 3.0) * (
            (z / (alpha - 1.0 / 3.0)) ** (1.0 / 3.0) - 1.0
        )
        return eps

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        alpha, beta = self.encode(input)
        z = self.reparameterize(alpha, beta)
        return [self.decode(z), input, alpha, beta]

    # def I_function(self, alpha_p, beta_p, alpha_q, beta_q):
    #     return - (alpha_q * beta_q) / alpha_p - \
    #            beta_p * torch.log(alpha_p) - torch.lgamma(beta_p) + \
    #            (beta_p - 1) * torch.digamma(beta_q) + \
    #            (beta_p - 1) * torch.log(alpha_q)
    def I_function(self, a, b, c, d):
        return (
            -c * d / a
            - b * torch.log(a)
            - torch.lgamma(b)
            + (b - 1) * (torch.digamma(d) + torch.log(c))
        )

    def vae_gamma_kl_loss(self, a, b, c, d):
        """
        https://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions
        b and d are Gamma shape parameters and
        a and c are scale parameters.
        (All, therefore, must be positive.)
        """

        a = 1 / a
        c = 1 / c
        losses = self.I_function(c, d, c, d) - self.I_function(a, b, c, d)
        return torch.sum(losses, dim=1)

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        alpha = args[2]
        beta = args[3]

        curr_device = input.device
        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = torch.mean(
            F.mse_loss(recons, input, reduction="none"), dim=(1, 2, 3)
        )

        # https://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions
        # alpha = 1./ alpha

        self.prior_alpha = self.prior_alpha.to(curr_device)
        self.prior_beta = self.prior_beta.to(curr_device)

        # kld_loss = - self.I_function(alpha, beta, self.prior_alpha, self.prior_beta)

        kld_loss = self.vae_gamma_kl_loss(
            alpha, beta, self.prior_alpha, self.prior_beta
        )

        # kld_loss = torch.sum(kld_loss, dim=1)

        loss = recons_loss + kld_loss
        loss = torch.mean(loss, dim=0)
        # print(loss, recons_loss, kld_loss)
        return {"loss": loss}  # , 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the modelSay
        :return: (torch.Tensor)
        """
        z = Gamma(self.prior_alpha, self.prior_beta).sample(
            (num_samples, self.latent_dim)
        )
        z = z.squeeze().to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


def init_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
