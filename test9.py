import os

import lightning as L
# import pandas as pd
# import seaborn as sn
import torch
import numpy as np
# from IPython.display import display
# from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
# from torchmetrics import Accuracy
# from torchvision import transforms
# from torchvision.datasets import MNIST
from typing import Dict, List
import torch.distributed as dist
from lightning.pytorch.callbacks import ModelCheckpoint

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
# BATCH_SIZE = 256 if torch.cuda.is_available() else 64
BATCH_SIZE = 16


class Dnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        # print(x.shape)
        # asdf
        return torch.relu(self.l1(x.view(x.size(0), -1)))


class LitModel(L.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.net(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.net(x), y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.net.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )
        '''
        # scheduler = LambdaLR(optimizer, self.lr_lambda_func)
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, 
            warmup_epochs=10, 
            max_epochs=10000, 
            warmup_start_lr=0.0, 
            eta_min=0.0, 
            last_epoch=-1
        )

        output_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
        '''
        return optimizer


class Mnist:
    def __init__(self):

        self.data = np.zeros((50000, 784))
        self.data[:, 0] = np.arange(50000)
        self.targets = np.ones(50000)

    def __getitem__(self, index):
        
        img, target = self.data[index].astype(np.float32), int(self.targets[index])
        # print(index)

        # Do augmentation here

        return img, target

    # def __len__(self) -> int:
    #     return len(self.data)


class Sampler:
    def __init__(self):
        pass

    def __iter__(self):
        for i in range(100):
            yield i

    def __len__(self):
        return 100


class BatchSampler:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):

        while True:
            yield range(self.batch_size)

    def __len__(self):
        return 100


class DistributedSamplerWrapper:
    def __init__(self, sampler: object) -> None:
        r"""Distributed wrapper of sampler.

        Args:
            sampler (Sampler object)

        Returns:
            None
        """

        self.sampler = sampler

    def __iter__(self) -> List:
        r"""Yield a part of mini-batch meta on each device.

        Args:
            None

        Returns:
            list_meta (List), a part of mini-batch meta.
        """

        if dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()

        else:
            num_replicas = 1
            rank = 0

        for list_meta in self.sampler:
            yield list_meta[rank :: num_replicas]

    def __len__(self) -> int:
        return len(self.sampler)



def add():
    
    batch_size_per_device = 16
    devices_num = torch.cuda.device_count()
    # devices_num = 1
    batch_size = batch_size_per_device * devices_num

    net = Dnn()

    # Init our model
    lit_model = LitModel(net=net)

    # Init DataLoader from MNIST Dataset
    train_dataset = Mnist()
    val_dataset = Mnist()

    # sampler = Sampler()
    batch_sampler = BatchSampler(batch_size=batch_size)
    batch_sampler = DistributedSamplerWrapper(batch_sampler)

    train_loader = DataLoader(
        dataset=train_dataset, 
        # batch_size=BATCH_SIZE, 
        # sampler=sampler,
        batch_sampler=batch_sampler, 
        # shuffle=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        dataset=train_dataset, 
        batch_sampler=batch_sampler, 
        num_workers=0,
    ) 

    checkpoint_callback1 = ModelCheckpoint(
        dirpath="./tmp",
        filename="{epoch}-{step}-{test_loss:.3f}",
        verbose=True,
        save_last=False,
        save_weights_only=True,
        every_n_train_steps=50,
        save_top_k=3,
        monitor="test_loss",
    )

    callbacks = [checkpoint_callback1]

    # Initialize a trainer
    trainer = L.Trainer(
        accelerator="auto",
        # accelerator="cpu",
        devices=devices_num,
        max_epochs=10,
        num_nodes=1,
        precision="32-true",
        callbacks=callbacks,
        # enable_checkpointing=True,
        use_distributed_sampler=False, 
    )

    # Train the model
    trainer.fit(
        model=lit_model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=None
    )


if __name__ == "__main__":
    add()