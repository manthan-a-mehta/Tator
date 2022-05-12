from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule, LightningModule
from torchvision.io import read_image
import torchvision.transforms as T
from pytorch_lightning.utilities.seed import seed_everything
import pandas as pd
from box import Box
import cv2
import matplotlib.pyplot as plt
import torch
from config import config
class FishDataset(Dataset):
    def __init__(self, df, image_size=224):
        self._X = df["thumbnail"].values
        self._y = None
        if "Fill Level" in df.keys():
            self._y = df["Fill Level"].values
        self._transform = T.Resize([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        image = self._transform(image)
        if self._y is not None:
            label = self._y[idx]
            return image, label,image_path
        return (image)

class FishDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            FishDataset(self._train_df, self._cfg.transform.image_size)
            if train
            else FishDataset(self._val_df, self._cfg.transform.image_size)
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)

# torch.autograd.set_detect_anomaly(True)

# df = pd.read_csv("merged.csv")
# config = Box(config)
# sample_dataloader = FishDataModule(df, df, config).val_dataloader()
# images, labels = iter(sample_dataloader).next()

# for it, (image, label) in enumerate(zip(images[:1], labels[:1])):
#     print(label)
#     plt.subplot(4, 4, it+1)
#     plt.imshow(image.permute(1, 2, 0))
#     plt.axis('off')
#     plt.title(f'Fill Level: {int(label)}')
#     plt.show()

# plt.show()
# df["thumbnail"] = df["thumbnail"].apply(lambda x: os.path.join(config.root, "train", x + ".jpg"))