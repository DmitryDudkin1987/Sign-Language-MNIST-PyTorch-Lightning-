from dataclasses import dataclass

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassFBetaScore,
)


@dataclass
class CFG:
    """Конфигурация обучения."""

    lr: float = 1e-3
    epochs: int = 30
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

        # --- метрики ---
        self.train_fbeta = MulticlassFBetaScore(
            beta=2.0, num_classes=self.n_classes, average="macro"
        )
        self.val_fbeta = MulticlassFBetaScore(
            beta=2.0, num_classes=self.n_classes, average="macro"
        )
        self.train_auroc = MulticlassAUROC(
            num_classes=self.n_classes, average="weighted"
        )
        self.val_auroc = MulticlassAUROC(
            num_classes=self.n_classes, average="weighted"
        )
        self.train_acc = MulticlassAccuracy(num_classes=self.n_classes)
        self.val_acc = MulticlassAccuracy(num_classes=self.n_classes)

        # --- сеть (без изменений) ---
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1,
                      stride=self.stride, dilation=self.dilation),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1,
                      stride=self.stride, dilation=self.dilation),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )
        self.lin1 = nn.Linear(in_features=16 * 7 * 7, out_features=100)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.lin2 = nn.Linear(100, self.n_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view((x.shape[0], -1))
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        return self.lin2(x)

    def basic_step(self, batch, batch_idx, step: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)

        if step == "train":
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/acc", self.train_acc(preds, y),
                     on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/fbeta", self.train_fbeta(preds, y),
                     on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/auroc", self.train_auroc(probs, y),
                     on_step=False, on_epoch=True, prog_bar=True)

        elif step == "valid":
            self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("valid/acc", self.val_acc(preds, y),
                     on_step=False, on_epoch=True, prog_bar=True)
            self.log("valid/fbeta", self.val_fbeta(preds, y),
                     on_step=False, on_epoch=True, prog_bar=True)
            self.log("valid/auroc", self.val_auroc(probs, y),
                     on_step=False, on_epoch=True, prog_bar=True)

        # Для test-ветки в примере логирования нет — можно добавить по аналогии,
        # но чтобы поведение совпало с примером, оставляю как есть.
        return loss

    def training_step(self, batch, batch_idx):
        return self.basic_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.basic_step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self.basic_step(batch, batch_idx, "test")

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
                "interval": "epoch",
                "frequency": 1,
                "monitor": None,
            },
        }