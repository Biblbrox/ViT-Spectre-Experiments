# %% Cell 1
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import sys

import lightning.pytorch as L

sys.path.append("../..")

import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from spectre_vit.configs.parser import parse_config
from spectre_vit.datasets.CIFAR100Dataset import CIFAR100DataModule
from spectre_vit.datasets.ImageNet1kDataset import ImageNet1kDataModule
from spectre_vit.datasets.MNISTDataset import MNISTDataModule
from spectre_vit.models.fnet.fnet import FNet
from spectre_vit.models.spectre.spectre import SpectreViT
from spectre_vit.models.vit.vit import ViT

# %%
# Read params from config
model_name = "spectre_vit"
dataset_name = "imagenet1k"
config_path = f"{model_name}/configs/{model_name}_{dataset_name}.py"
c = parse_config(config_path)
experiment_name = f"{model_name}_{c.num_heads}h_hid{c.hidden_dim}_emb{c.embed_dim}_patch{c.patch_size}_enc{c.num_encoders}_{dataset_name}"

random.seed(c.random_seed)
np.random.seed(c.random_seed)
torch.manual_seed(c.random_seed)
torch.cuda.manual_seed(c.random_seed)
torch.cuda.manual_seed_all(c.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Test ViT
if model_name == "spectre_vit":
    model = SpectreViT(c).to(device)
elif model_name == "vit":
    model = ViT(c).to(device)
elif model_name == "fnet":
    model = FNet(c).to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# %%
if dataset_name == "cifar100":
    datamodule = CIFAR100DataModule(c)
elif dataset_name == "imagenet1k":
    datamodule = ImageNet1kDataModule(c)
elif dataset_name == "mnist":
    datamodule = MNISTDataModule(c)

# %%
use_amp = True
logger = TensorBoardLogger("../../runs/", name=experiment_name, version=1)
trainer = L.Trainer(accelerator="gpu", devices=1, logger=logger, max_epochs=c.epochs)
trainer.fit(model, datamodule=datamodule)

# %%
