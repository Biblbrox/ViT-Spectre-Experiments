# %% Cell 1
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import sys
import timeit

import lightning.pytorch as L

sys.path.append("../..")

import numpy as np
import torch
from lightning.pytorch.loggers import LitLogger, TensorBoardLogger
from torch import nn, optim

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
experiment_name = f"{model_name}_{c.num_heads}h_hid{c.hidden_dim}_emb{c.embed_dim}_patch{c.patch_size}_enc{c.num_encoders}"

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
use_amp = False
T = 2
soft_target_loss_weight = 0.25
ce_loss_weight = 0.75

criterion = nn.CrossEntropyLoss()
# criterion_dist = nn.KLDivLoss(reduction="batchmean")
criterion_dist = nn.CosineSimilarity()
# optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)
optimizer = optim.AdamW(
    model.parameters(), betas=c.adam_betas, lr=c.learning_rate, weight_decay=c.adam_weight_decay
)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
start = timeit.default_timer()
for epoch in range(c.epochs):
    model.train()
    teacher.eval()
    train_labels = []
    train_preds = []
    train_running_loss = 0

    for idx, data in enumerate(train_dataloader):
        img = data["img_model"].float().to(device)
        img_teacher = data["img_teacher"].float().to(device)
        label = data["label"].type(torch.uint8).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            student_logits, student_feat = model(img, return_features=True)
            with torch.no_grad():
                teacher_logits, teacher_feat = teacher(img_teacher, return_features=True)

            y_pred_label = torch.argmax(student_logits, dim=1)

            train_labels.extend(label.cpu().detach())
            train_preds.extend(y_pred_label.cpu().detach())

            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            soft_targets_loss = (
                torch.sum(soft_targets * (soft_targets.log() - soft_prob))
                / soft_prob.size()[0]
                * (T**2)
            )

            # student_feat = F.normalize(student_feat, dim=-1)
            # teacher_feat = F.normalize(teacher_feat, dim=-1)

            # loss_dist = 1 - criterion_dist(student_feat, teacher_feat).mean()
            loss_ce = criterion(student_logits, label)
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * loss_ce

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("Batch Loss/Train", loss.item(), epoch * len(train_dataloader) + idx)
        writer.add_scalar(
            "Batch Loss/Dist", soft_targets_loss.item(), epoch * len(train_dataloader) + idx
        )
        writer.add_scalar("Batch Loss/CE", loss_ce.item(), epoch * len(train_dataloader) + idx)

        train_running_loss += loss.item()

    train_loss = train_running_loss / (idx + 1)

    model.eval()
    val_labels = []
    val_preds = []
    val_running_loss = 0
    with torch.no_grad():
        for idx, img_label in enumerate(val_dataloader):
            img = img_label[0].float().to(device)
            label = img_label[1].type(torch.uint8).to(device)

            student_logits = model(img)
            y_pred_label = torch.argmax(student_logits, dim=1)

            val_labels.extend(label.cpu().detach())
            val_preds.extend(y_pred_label.cpu().detach())

            loss = criterion(student_logits, label)
            val_running_loss += loss.item()

        val_loss = val_running_loss / (idx + 1)

        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        writer.add_scalar(
            "Accuracy/Train",
            sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels),
            epoch + 1,
        )

    stop = timeit.default_timer()
    writer.add_scalar("Training time", stop - start)
    writer.close()
    print(f"Training time: {stop - start:.2f}")
