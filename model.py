from dataclasses import dataclass

import torch
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F  # noqa: F401  (оставлен для совместимости)


@dataclass
class CFG:
    """Конфигурация обучения."""

    lr: float = 1e-3
    epochs: int = 5
    n_classes: int = 26


cfg = CFG()


class LightningModel(LightningModule):
    """Простая свёрточная сеть для классификации Sign Language MNIST."""

    def __init__(self, cfg: CFG):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.lr = cfg.lr

        self.stride = 1
        self.dilation = 1
        self.n_classes = cfg.n_classes

        self.block1 = nn.Sequential(
            # input=(batch, 1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1,
                stride=self.stride,
                dilation=self.dilation,
            ),
            nn.BatchNorm2d(8),
            # (batch, 8, 28, 28)
            nn.AvgPool2d(2),
            # (batch, 8, 14, 14)
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1,
                stride=self.stride,
                dilation=self.dilation,
            ),
            nn.BatchNorm2d(16),
            # (batch, 16, 14, 14)
            nn.AvgPool2d(2),
            # (batch, 16, 7, 7)
            nn.ReLU(),
        )

        self.lin1 = nn.Linear(in_features=16 * 7 * 7, out_features=100)
        # (batch, 100)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.lin2 = nn.Linear(100, self.n_classes)
        # (batch, 26)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view((x.shape[0], -1))
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.lin2(x)
        return x

    def basic_step(self, batch, batch_idx, step: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        on_step = step == "train"
        on_epoch = step != "train"

        self.log(
            f"{step}/loss",
            loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=on_step,
        )
        self.log(
            f"{step}/acc",
            acc,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.basic_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.basic_step(batch, batch_idx, "valid")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.basic_step(batch, batch_idx, "test")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=0.05
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # шаг планировщика раз в эпоху
                "frequency": 1,
                "monitor": None,  # для CosineAnnealingLR не нужен
            },
        }