#!/usr/bin/env python

import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse
import random
import sys

import torch
from lightning import Trainer

from data import URL_TEST, URL_TRAIN, SignLanguageLightning
from model import CFG, LightningModel
from utils import str2bool


cfg = CFG()


def main(fast_dev_run: bool) -> None:
    datamodule = SignLanguageLightning(URL_TRAIN, URL_TEST)
    model = LightningModel(cfg)

    if fast_dev_run:
        try:
            probe = Trainer(
                max_epochs=cfg.epochs,
                fast_dev_run=True,
                enable_checkpointing=False,
                logger=False,
            )
            probe.fit(model, datamodule=datamodule)
            print("Тестовый прогон успешно пройден")
        except Exception as e:  # noqa: BLE001
            print(f"Тестовый прогон завершился с ошибкой: {e}")
            sys.exit(1)
        return

    trainer = Trainer(
        max_epochs=cfg.epochs,
        log_every_n_steps=1,
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint("models/myconvnet_sign_lang.ckpt")

    model.eval()
    with torch.no_grad():
        idx = random.randrange(len(datamodule.val_dataset))
        x, y = datamodule.val_dataset[idx]

        logits = model(x.unsqueeze(0))  # (1, C, H, W)
        probs = torch.softmax(logits, dim=1)[0]  # (C,)
        pred = logits.argmax(dim=1).item()
        conf = probs[pred].item()

    print(f"Индекс образца: {idx}")
    print(f"Истинная метка: {y}")
    print(f"Предсказание:   {pred}")
    print(f"Уверенность:    {conf:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", type=str2bool, default=False)
    args = parser.parse_args()
    main(args.fast_dev_run)
