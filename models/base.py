import pytorch_lightning as pl
import torch
from typing import Any, List

from abc import abstractmethod, ABC

class BaseVAE(pl.LightningModule):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> torch.Tensor:
        raise RuntimeWarning()

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        results = self.forward(input=batch[0], labels=batch[1])
        train_loss = self.model.loss_function(*results,
                                              M_N = batch.shape[0] / self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        for key, val in train_loss.items():
            self.log(key, val)

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        results = self.forward(input=batch[0], labels=batch[1])
        val_loss = self.model.loss_function(*results,
                                            M_N = batch.shape[0] / self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss
