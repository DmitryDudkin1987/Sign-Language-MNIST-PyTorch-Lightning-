# Sign Language MNIST (PyTorch Lightning)

Учебный проект: классификация жестов ASL по изображениям 28×28 на датасете Sign Language MNIST.

## Структура

PyTourch Lightening/
├── main.py           # точка входа
├── data.py           # Dataset + LightningDataModule
├── model.py          # CFG + LightningModel
├── utils.py          # str2bool
├── requirements.txt
├── data/             # скачивается автоматически
├── checkpoints/      # лучшие чекпоинты (создать заранее)
└── models/           # финальный чекпоинт (создать заранее)

## Требования

- Python 3.10–3.12 (3.13 может не найти колёса для torch)
- ~1.5 GB места, интернет при первом запуске

## Установка

python -m venv .venv
.venv\Scripts\activate.bat          # Windows (cmd)
# source .venv/bin/activate          # Linux/macOS

python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir models checkpoints

## Запуск

# быстрая проверка пайплайна (~1 мин)
python main.py --fast_dev_run true

# полное обучение (до 30 эпох, с EarlyStopping)
python main.py

Аргумент --fast_dev_run принимает true/false/yes/no.

## Что происходит

1. Скачивает train.zip и test.zip в ./data/, распаковывает CSV.
2. Делит train 80/20 (stratify по label), train — с аугментациями.
3. Обучает CNN до 30 эпох (AdamW + CosineAnnealingLR) с callbacks:
   - EarlyStopping(monitor="valid/loss", patience=3) — останавливает обучение,
     если валидационная потеря не улучшается 3 эпохи подряд;
   - ModelCheckpoint(monitor="valid/loss", dirpath="checkpoints/",
     filename="best-{epoch}", save_top_k=1) — сохраняет лучшую модель
     по valid/loss в checkpoints/best-{epoch}.ckpt.
4. Сохраняет финальный чекпоинт models/myconvnet_sign_lang.ckpt.
5. Печатает предсказание на случайном val-примере.

## Архитектура

(B,1,28,28)
 → Conv2d(1→8)   + BN + AvgPool2d(2) + ReLU      → (B,8,14,14)
 → Conv2d(8→16)  + BN + AvgPool2d(2) + ReLU      → (B,16,7,7)
 → Flatten → Linear(784→100) → LeakyReLU → Dropout(0.3) → Linear(100→26)

Аугментации (только train): RandomHorizontalFlip(p=0.1), RandomApply([RandomRotation(±180°)], p=0.2).

## Метрики

Логируются через torchmetrics:

| Метрика | Описание |
|---|---|
| train/loss, valid/loss | CrossEntropyLoss |
| train/acc, valid/acc | MulticlassAccuracy |
| train/fbeta, valid/fbeta | MulticlassFBetaScore (β=2, macro) |
| train/auroc, valid/auroc | MulticlassAUROC (weighted) |

EarlyStopping и ModelCheckpoint мониторят valid/loss.

## Ожидаемые метрики

| Метрика | Значение |
|---|---|
| valid/acc | ~0.90–0.99 |
| valid/fbeta (macro, β=2) | ~0.88–0.98 |
| valid/auroc (weighted) | ~0.99+ |
| Время CPU | ~15–40 мин (с EarlyStopping обычно 7–15 эпох) |
| Время GPU | ~3–8 мин |

## Частые ошибки

| Ошибка | Решение |
|---|---|
| No module named 'lightning' | Активируй venv, pip install -r requirements.txt |
| No module named 'torchmetrics' | pip install -r requirements.txt (обновлённый) |
| Could not find a version... torch | Поставь Python 3.11/3.12 |
| FileNotFoundError: models/...ckpt | mkdir models checkpoints |
| BrokenPipeError / DataLoader worker | Поставь num_workers=0 в data.py |
| OSError: [WinError 10054] | Файрвол режет urllib — скачай zip вручную в ./data/ |
| Checkpoint directory checkpoints/ does not exist | mkdir checkpoints |
| «Невозможно запустить это приложение» | Это не python.exe, а другой .exe — запускай через терминал |

