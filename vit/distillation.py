import torch.nn as nn
import torch


class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes, embed_dim=384):
        super(DinoClassifier, self).__init__()
        self.backbone = backbone
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, return_features=False):
        feats = self.backbone.forward_features(x)
        cls_token = feats['x_norm_clstoken']  # [B, C]

        feats = cls_token  # [B, C]
        logits = self.decoder(feats)
        if return_features:
            return logits, feats
        else:
            return logits


class DistillationDatasetCls(torch.utils.data.Dataset):
    def __init__(self, samples, teacher_tf, model_tf):
        self.samples = samples
        self.teacher_tf = teacher_tf
        self.model_tf = model_tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]

        img1 = self.model_tf(img)
        img2 = self.teacher_tf(img)
        return {
            "img_teacher": img2,
            "img_model": img1,
            "label": label
        }
