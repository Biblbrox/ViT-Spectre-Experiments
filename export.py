# %% Cell 1
import onnx
import torch
from onnxsim import simplify

from spectre_vit.configs.parser import parse_config
from spectre_vit.models.spectre.spectre import SpectreViT

# %% Cell 2
# Model configuration
config_path = "spectre_vit/configs/spectre_vit_cifar100.py"
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


# %% Cell 3
def export_model(model_name, weights_path=None):
    model = (
        SpectreViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_encoders=num_encoders,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            method=method,
        )
        .cuda()
        .eval()
    )

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, weights_only=True), strict=True)

    print(f"Exporting model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32).cuda()

    torch.onnx.export(
        model,
        (input_tensor,),
        f"export/{model_name}.onnx",
        opset_version=22,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
        external_data=False,
    )
    model = onnx.load(f"export/{model_name}.onnx")
    model_simp, check = simplify(model)
    onnx.save(
        model_simp,
        f"export/{model_name}.onnx",
    )

    assert check, "Simplified ONNX model could not be validated"


model_name = f"spectre_vit_mixing_{num_heads}h_hid{hidden_dim}_emb{embed_dim}_patch{patch_size}_enc{num_encoders}"
model_weights = f"runs/{model_name}/model_best.pt"
export_model(model_name, model_weights)
