import io
from os.path import join

import litdata as ld
import torch
from PIL import Image
from torchvision.transforms import v2

from spectre_vit.datasets.DefaultDataset import DefaultDataset


class ImageNet1kDataModule(DefaultDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_uri = "hf://datasets/ILSVRC/imagenet-1k"
        self.train_transform = v2.Compose([
            v2.PILToTensor(),
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
            v2.RandomGrayscale(p=0.2),
            v2.RandomAffine(30),
            v2.RandomApply([v2.GaussianBlur(3)]),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
            v2.RandomErasing(0.5, inplace=True),
        ])

        self.val_transform = v2.Compose([
            v2.PILToTensor(),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ])

    def prepare_data(self):
        # download
        train_uri = "data/train-*.parquet"
        val_uri = "data/validation-*.parquet"
        cache_dir = join(self.cache_dir, "imagenet1k")
        self.train_ds = ld.StreamingDataset(
            f"{self.dataset_uri}/{train_uri}", index_path=self.index_path, cache_dir=cache_dir
        )
        self.val_ds = ld.StreamingDataset(
            f"{self.dataset_uri}/{val_uri}", index_path=self.index_path, cache_dir=cache_dir
        )

    def collate(self, batch, mode):
        """
        Custom collate function for PyTorch DataLoader with resizing.

        Args:
            batch (list of tuples): Each element is (image_dict, label).

        Returns:
            inputs (Tensor): Batched inputs (B, C, H, W).
            labels (Tensor): Batched labels (B,).
        """
        imgs = []
        labels = []

        for img_dict in batch:
            image = Image.open(io.BytesIO(img_dict["image"]["bytes"]))
            if mode == "train":
                tensor_img = self.train_transform(image)
            else:
                tensor_img = self.val_transform(image)
            imgs.append(tensor_img)
            labels.append(img_dict["label"])

        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)
