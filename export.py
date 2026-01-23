# %% Cell 1
from spectre_vit.base_vit import ViT
from configs.parser import parse_config
import torch
# %% Cell 2
# Model configuration
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


# %% Cell 3


def export_model(model_name):
    model = ViT(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                num_classes=num_classes, embed_dim=embed_dim, num_encoders=num_encoders,
                num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout, activation=activation,
                method=method).cuda()
    print(
        f"Exporting model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    torch.save(model, f"export/{model_name}.pt")
    input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32).cuda()
    torch.onnx.export(model, (input_tensor, ), f"export/{model_name}.onnx", input_names=[
                      "input"], output_names=["output"], dynamo=True)


model_name = f"spectre_vit{img_size}_HID{hidden_dim}_HEAD{num_heads}_ENC{num_encoders}"
export_model(model_name)
