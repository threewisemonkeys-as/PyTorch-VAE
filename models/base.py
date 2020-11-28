from abc import abstractmethod
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch


class BaseVAE(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float = 0,
        scheduler_gamma: Optional[float] = None,
        submodel: Optional[torch.nn.Module] = None,
        submodel_lr: Optional[float] = None,
        submodel_weight_decay: Optional[float] = None,
        submodel_scheduler_gamma: Optional[float] = None,
    ) -> None:
        super(BaseVAE, self).__init__()

        if submodel is not None and submodel_lr is None:
            raise ValueError("Learning rate of submodel not specified")

        if (
            submodel_lr is not None
            or submodel_weight_decay is not None
            or submodel_scheduler_gamma is not None
        ) and submodel is None:
            raise ValueError("Submodel not given")

        self.submodel = submodel
        self.optim_param = {
            "lr": lr,
            "weight_decay": weight_decay,
            "scheduler_gamme": scheduler_gamma,
            "submodel_lr": submodel_lr,
            "submodel_weight_decay": submodel_weight_decay,
            "submodel_scheduler_gamma": submodel_scheduler_gamma,
        }

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        results = self.forward(input=batch[0], labels=batch[1])
        train_loss = self.model.loss_function(
            *results,
            M_N=batch.shape[0] / self.num_train_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        for key, val in train_loss.items():
            self.log(key, val)

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        results = self.forward(input=batch[0], labels=batch[1])
        val_loss = self.model.loss_function(
            *results,
            M_N=batch.shape[0] / self.num_val_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        return val_loss

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.optim_params["lr"],
            weight_decay=self.optim_params["weight_decay"],
        )
        optims.append(optimizer)

        if self.optim_param["scheduler_gamma"] is not None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optims[0], gamma=self.optim_param["scheduler_gamma"]
            )
            scheds.append(scheduler)

        # Check if more than 1 optimizer is required (Used for adversarial training)
        if self.optim_param["submodel_lr"] is not None:
            optimizer2 = torch.optim.Adam(
                self.submodel.parameters(),
                lr=self.optim_param["submodel_lr"],
            )
            optims.append(optimizer2)

            if self.optim_param["submodel_scheduler_gamma"] is not None:
                scheduler2 = torch.optim.lr_scheduler.ExponentialLR(
                    optims[1], gamma=self.optim_param["submodel_scheduler_gamma"]
                )
                scheds.append(scheduler2)

        return optims, scheds
