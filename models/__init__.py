from .beta_vae import BetaVAE
from .betatc_vae import BetaTCVAE
from .cat_vae import CategoricalVAE
from .cvae import ConditionalVAE
from .dfcvae import DFCVAE
from .dip_vae import DIPVAE
from .fvae import FactorVAE
from .gamma_vae import GammaVAE
from .hvae import HVAE
from .info_vae import InfoVAE
from .iwae import IWAE
from .joint_vae import JointVAE
from .logcosh_vae import LogCoshVAE
from .lvae import LVAE
from .miwae import MIWAE
from .mssim_vae import MSSIMVAE
from .swae import SWAE
from .twostage_vae import TwoStageVAE
from .vampvae import VampVAE
from .vanilla_vae import VanillaVAE
from .vq_vae import VQVAE
from .wae_mmd import WAE_MMD

# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE
GumbelVAE = CategoricalVAE
