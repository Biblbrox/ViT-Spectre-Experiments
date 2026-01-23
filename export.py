# %% Cell 1
from spectre_vit.base_vit import ViT
import torch
# %% Cell 2
# Model configuration
BATCH_SIZE = 16
EPOCHS = 1000
LEARNING_RATE = 1e-3
NUM_CLASSES = 100
PATCH_SIZE = 4
IMG_SIZE = 32
IN_CHANNELS = 3
NUM_HEADS = 8
DROPOUT = 0.001
HIDDEN_DIM = 256
ADAM_WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.9, 0.999)
ACTIVATION = 'gelu'
NUM_ENCODERS = 4
# EMBED_DIM = (PATCH_SIZE**2) * IN_CHANNELS # 16
EMBED_DIM = 512
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 49
USE_SPECTRE = True
SPECTRE_THRESHOLD = 1.0
METHOD = 'fft_mh_spectrelayers'
# %% Cell 3


def export_model(model_name):
    model = ViT(img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS,
                num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, num_encoders=NUM_ENCODERS,
                num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, activation=ACTIVATION,
                method=METHOD).cuda()
    print(
        f"Exporting model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    torch.save(model, f"export/{model_name}.pt")
    input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32).cuda()
    torch.onnx.export(model, (input_tensor, ), f"export/{model_name}.onnx", input_names=[
                      "input"], output_names=["output"], dynamo=True)


model_name = f"spectre_vit{IMG_SIZE}_HID{HIDDEN_DIM}_HEAD{NUM_HEADS}_ENC{NUM_ENCODERS}"
export_model(model_name)
