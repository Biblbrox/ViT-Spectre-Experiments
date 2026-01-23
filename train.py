# %% Cell 1
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import cv2
import numpy as np
from PIL import Image
# import polars as pd
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import random
import torch.nn.functional as F
import pytorch_warmup as warmup
import torchvision
import timeit
from tqdm import tqdm
from spectre_vit.base_vit import ViT
from spectre_vit.distillation import DinoClassifier
from dinov3.models.vision_transformer import DinoVisionTransformer
from torch.utils.tensorboard import SummaryWriter
from spectre_vit.distillation import DistillationDatasetCls
from configs.parser import parse_config
# %%
# Read params from config
config_path = "configs/spectre_vit.py"
config = parse_config(config_path)
random_seed = config.random_seed
batch_size = config.batch_size
epochs = config.epochs
learning_rate = config.learning_rate
num_classes = config.num_classes
patch_size = config.patch_size
img_size = config.img_size
in_channels = config.in_channels
num_heads = config.num_heads
dropout = config.dropout
hidden_dim = config.hidden_dim
adam_weight_decay = config.adam_weight_decay
adam_betas = config.adam_betas
activation = config.activation
num_encoders = config.num_encoders
embed_dim = config.embed_dim
num_patches = config.num_patches
use_spectre = config.use_spectre
spectre_threshold = config.spectre_threshold
method = config.method
experiment_name = "spectre_vit_fftmh16_spectrelayers_fusedheads"
use_distillation = False

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter(f'runs/{experiment_name}')
torch.serialization.add_safe_globals([DinoClassifier])
torch.serialization.add_safe_globals([DinoVisionTransformer])

# %%
# Test ViT
model = ViT(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            num_classes=num_classes, embed_dim=embed_dim, num_encoders=num_encoders,
            num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout, activation=activation,
            method=method).to(device)
if use_distillation:
    current_dir = os.path.curdir
    DINO_REPO = f"{current_dir}/dinov3"
    BACKBONE_PATH = "/storage/experiments-ml/weights/dino/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    dino = torch.hub.load(DINO_REPO, 'dinov3_vits16', source='local',
                          weights=BACKBONE_PATH, skip_validation=True, map_location='cuda')
    dino.cuda().eval()
    # Freeze all parameters in the backbone
    for param in dino.parameters():
        param.requires_grad = False

    teacher = DinoClassifier(
        backbone=dino,
        num_classes=num_classes).to(device)
    teacher.load_state_dict(torch.load(
        './checkpoints/dinov3s16_best.pth', weights_only=True), strict=False)
    teacher.train()

# Print model params number
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# writer.add_graph(model, torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE).to(device))

# %%
# Transforms
if use_distillation:
    transform_dino = transforms.Compose([
        transforms.Resize(
            256, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        ),
    ])

train_transform_spectre = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomAffine(30),
    transforms.RandomApply([transforms.GaussianBlur(3)]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
    transforms.RandomErasing(0.5, inplace=True),
])

eval_transform_spectre = transforms.Compose([
    # transforms.Resize(
    #    256, interpolation=transforms.InterpolationMode.BICUBIC
    # ),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
])

# Load CIFAR100 dataset
val_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=eval_transform_spectre)
test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=eval_transform_spectre)
if use_distillation:
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=None)
    train_dataset = DistillationDatasetCls(
        samples=train_dataset, teacher_tf=transform_dino, model_tf=train_transform_spectre)
else:
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform_spectre)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Display sample images
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    if use_distillation:
        data = train_dataset[i]
        img = data['img_model']
        label = data['label']
    else:
        img, label = train_dataset[i]
    img = img.permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5  # Unnormalize
    ax.imshow(img)
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# %%
if not use_distillation:
    use_amp = False

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY, nesterov=True, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(
    ), betas=adam_betas, lr=learning_rate, weight_decay=adam_weight_decay)
    num_steps = len(train_dataloader) * epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps)
    warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start = timeit.default_timer()
    for epoch in range(epochs):
        model.train()
        train_labels = []
        train_preds = []
        train_running_loss = 0
        spectre_running_loss = 0

        for idx, img_label in enumerate(train_dataloader):
            img = img_label[0].float().to(device)
            label = img_label[1].type(torch.uint8).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                y_pred = model(img)
                y_pred_label = torch.argmax(y_pred, dim=1)

                train_labels.extend(label.cpu().detach())
                train_preds.extend(y_pred_label.cpu().detach())

                loss = criterion(y_pred, label)

                # Find spectre loss in all spectre layers
                spectre_loss = 0
                for module in model.modules():
                    if hasattr(module, 'spectre_loss'):
                        spectre_loss += module.spectre_loss * 0.1
                loss += spectre_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with warmup_scheduler.dampening():
                lr_scheduler.step()

            train_running_loss += loss.item()
            spectre_running_loss += spectre_loss

        train_loss = train_running_loss / (idx + 1)
        spectre_loss = spectre_running_loss / (idx + 1)

        model.eval()
        val_labels = []
        val_preds = []
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_label in enumerate(val_dataloader):
                img = img_label[0].float().to(device)
                label = img_label[1].type(torch.uint8).to(device)

                y_pred = model(img)
                y_pred_label = torch.argmax(y_pred, dim=1)

                val_labels.extend(label.cpu().detach())
                val_preds.extend(y_pred_label.cpu().detach())

                loss = criterion(y_pred, label)
                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)
            writer.add_scalar('Loss/Train', train_loss, epoch + 1)
            writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
            writer.add_scalar('Accuracy/Train', sum(1 for x, y in zip(train_preds,
                              train_labels) if x == y) / len(train_labels), epoch + 1)
            writer.add_scalar('Accuracy/Validation', sum(1 for x, y in zip(
                val_preds, val_labels) if x == y) / len(val_labels), epoch + 1)

    stop = timeit.default_timer()
    writer.add_scalar('Training time', stop - start)
    writer.close()
    print(f"Training time: {stop - start:.2f}")

# %%
if use_distillation:
    use_amp = False
    T = 2
    soft_target_loss_weight = 0.25
    ce_loss_weight = 0.75

    criterion = nn.CrossEntropyLoss()
    # criterion_dist = nn.KLDivLoss(reduction="batchmean")
    criterion_dist = nn.CosineSimilarity()
    # optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)
    optimizer = optim.AdamW(model.parameters(
    ), betas=adam_betas, lr=learning_rate, weight_decay=adam_weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start = timeit.default_timer()
    for epoch in range(epochs):
        model.train()
        teacher.eval()
        train_labels = []
        train_preds = []
        train_running_loss = 0

        for idx, data in enumerate(train_dataloader):
            img = data['img_model'].float().to(device)
            img_teacher = data['img_teacher'].float().to(device)
            label = data['label'].type(torch.uint8).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                student_logits, student_feat = model(img, return_features=True)
                with torch.no_grad():
                    teacher_logits, teacher_feat = teacher(
                        img_teacher, return_features=True)

                y_pred_label = torch.argmax(student_logits, dim=1)

                train_labels.extend(label.cpu().detach())
                train_preds.extend(y_pred_label.cpu().detach())

                soft_targets = nn.functional.softmax(
                    teacher_logits / T, dim=-1)
                soft_prob = nn.functional.log_softmax(
                    student_logits / T, dim=-1)

                soft_targets_loss = torch.sum(
                    soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

                # student_feat = F.normalize(student_feat, dim=-1)
                # teacher_feat = F.normalize(teacher_feat, dim=-1)

                # loss_dist = 1 - criterion_dist(student_feat, teacher_feat).mean()
                loss_ce = criterion(student_logits, label)
                loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * loss_ce

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar('Batch Loss/Train', loss.item(),
                              epoch * len(train_dataloader) + idx)
            writer.add_scalar(
                'Batch Loss/Dist', soft_targets_loss.item(), epoch * len(train_dataloader) + idx)
            writer.add_scalar('Batch Loss/CE', loss_ce.item(),
                              epoch * len(train_dataloader) + idx)

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

            writer.add_scalar('Loss/Train', train_loss, epoch + 1)
            writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
            writer.add_scalar('Accuracy/Train', sum(1 for x, y in zip(train_preds,
                              train_labels) if x == y) / len(train_labels), epoch + 1)
            writer.add_scalar('Accuracy/Validation', sum(1 for x, y in zip(
                val_preds, val_labels) if x == y) / len(val_labels), epoch + 1)

    stop = timeit.default_timer()
    writer.add_scalar('Training time', stop - start)
    writer.close()
    print(f"Training time: {stop - start:.2f}")
