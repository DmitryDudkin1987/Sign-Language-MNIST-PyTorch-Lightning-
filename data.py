import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import torch
import torch.utils.data as torch_data
import torchvision.transforms as transforms
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split


URL_TRAIN = (
    "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/"
    "data/sign_mnist_train.csv.zip"
)
URL_TEST = (
    "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/"
    "data/sign_mnist_test.csv.zip"
)


class SignLanguageDataset(torch_data.Dataset):
    """Датасет Sign Language MNIST."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int):
        label = self.df.iloc[index, 0]
        img = self.df.iloc[index, 1:].values.reshape(28, 28)
        img = torch.Tensor(img).unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class SignLanguageLightning(LightningDataModule):
    """LightningDataModule для Sign Language MNIST."""

    def __init__(
        self,
        url_train: str,
        url_test: str,
        batch_size: int = 64,
        num_workers: int = 2,
    ) -> None:
        super().__init__()
        self.url_train = url_train
        self.url_test = url_test
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=(-180, 180))], p=0.2
                ),
            ]
        )

    def prepare_data(self) -> None:
        data_dir = Path("./data")
        print(data_dir.resolve())
        data_dir.mkdir(parents=True, exist_ok=True)

        datasets = [
            (
                self.url_train,
                "train.zip",
                "sign_mnist_train",
                "sign_mnist_train.csv",
            ),
            (
                self.url_test,
                "test.zip",
                "sign_mnist_test",
                "sign_mnist_test.csv",
            ),
        ]

        for url, zip_name, extract_name, target_name in datasets:
            zip_path = data_dir / zip_name
            if not zip_path.exists():
                print(f"Downloading {url} -> {zip_path}")
                urllib.request.urlretrieve(url, zip_path)
            else:
                print(f"Already exists: {zip_path}")

            extract_dir = data_dir / extract_name
            target_file = extract_dir / target_name

            if not target_file.exists():
                print(f"Extracting {zip_path} -> {extract_dir}")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)
            else:
                print(f"Already extracted: {target_file}")

    def setup(self, stage=None) -> None:
        data_dir = Path("./data")
        train_csv = data_dir / "sign_mnist_train" / "sign_mnist_train.csv"
        test_csv = data_dir / "sign_mnist_test" / "sign_mnist_test.csv"

        if stage in (None, "fit"):
            self.train_df = pd.read_csv(train_csv)
            train_df, val_df = train_test_split(
                self.train_df,
                test_size=0.2,
                random_state=42,
                shuffle=True,
                stratify=self.train_df["label"],
            )
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)

            self.train_dataset = SignLanguageDataset(
                train_df, transform=self.transform
            )
            self.val_dataset = SignLanguageDataset(val_df)

        if stage in (None, "test", "predict"):
            self.test_df = pd.read_csv(test_csv)
            self.test_dataset = SignLanguageDataset(self.test_df)

    def train_dataloader(self):
        return torch_data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch_data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch_data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )