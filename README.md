# Sign Language MNIST (PyTorch Lightning)

Учебный проект: классификация жестов ASL по изображениям 28×28 на датасете Sign Language MNIST.

## Структура

```
PyTourch Lightening/
├── main.py           # точка входа
├── data.py           # Dataset + LightningDataModule
├── model.py          # CFG + LightningModel
├── utils.py          # str2bool
├── requirements.txt
├── data/             # скачивается автоматически
└── models/           # чекпоинты (создать заранее)
```

## Требования

- Python 3.10–3.12 (3.13 может не найти колёса для `torch`)
- ~1.5 GB места, интернет при первом запуске

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate.bat          # Windows (cmd)
# source .venv/bin/activate          # Linux/macOS

python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir models
```

## Запуск

```bash
# быстрая проверка пайплайна (~1 мин)
python main.py --fast_dev_run true

# полное обучение (5 эпох)
python main.py
```

Аргумент `--fast_dev_run` принимает `true`/`false`/`yes`/`no`.

## Что происходит

1. Скачивает `train.zip` и `test.zip` в `./data/`, распаковывает CSV.
2. Делит train 80/20 (stratify по `label`), train — с аугментациями.
3. Обучает CNN 5 эпох (`AdamW` + `CosineAnnealingLR`).
4. Сохраняет `models/myconvnet_sign_lang.ckpt`.
5. Печатает предсказание на случайном val-примере.

## Архитектура

```
(B,1,28,28)
 → Conv2d(1→8)   + BN + AvgPool2d(2) + ReLU      → (B,8,14,14)
 → Conv2d(8→16)  + BN + AvgPool2d(2) + ReLU      → (B,16,7,7)
 → Flatten → Linear(784→100) → LeakyReLU → Dropout(0.3) → Linear(100→26)
```

Аугментации (только train): `RandomHorizontalFlip(p=0.1)`, `RandomApply([RandomRotation(±180°)], p=0.2)`.

## Ожидаемые метрики

| Метрика | Значение |
|---|---|
| val/acc | ~0.90–0.98 |
| Время CPU | ~5–15 мин |
| Время GPU | ~1–2 мин |

## Частые ошибки

| Ошибка | Решение |
|---|---|
| `No module named 'lightning'` | Активируй venv, `pip install -r requirements.txt` |
| `Could not find a version... torch` | Поставь Python 3.11/3.12 |
| `FileNotFoundError: models/...ckpt` | `mkdir models` |
| `BrokenPipeError` / DataLoader worker | Поставь `num_workers=0` в `data.py` |
| `OSError: [WinError 10054]` | Файрвол режет `urllib` — скачай zip вручную в `./data/` |
| «Невозможно запустить это приложение» | Это не `python.exe`, а другой `.exe` — запускай через терминал |

## Лицензия
