# Spectre ViT experiment project

## Content

- vit - python pytorch experiments folder
- inference - rust implementation of real-time inference.

### Experiments setup
In order to build the project you should the following:
- Install uv if it's not already installed: 
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
- Synchronize the environment and build all dependencies:
```bash
uv sync
```

Now you should be able to choose spectre_vit environment in every jupyter notebook.

### Notebooks
- vit_spectre_cifar100.ipynb - Cifar100 training pipeline;
- vit_spectre_dino_pretraining.ipynb - DINOv3 fine-tuning pipeline;
- vit_spectre_mnist.ipynb - MNIST100 traning pipeline;
- fft_experiments.ipynb - Visualization of fft features from the model;
- dwt_experiments.ipynb - Visualization of dwt features from the model.


### Inference setup
Make sure you installed arm64 cross-compiler:
```
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

Export pkg-config sysroot variable to your aarch64 sysroot:
```
export PKG_CONFIG_SYSROOT_DIR="/usr/aarch64-linux-gnu"
```

Run cargo build for the specified architecture:
```
cargo build --release --target aarch64-unknown-linux-gnu
```

If you want to build for x64, omit --target keyword:
```
cargo build --release
```

To specify a required glibc version edit file glibc.version and run cargo with the following variable:
```
RUSTFLAGS="-C link-arg=-Wl,--version-script=./glibc.version" cargo build --release --target aarch64-unknown-linux-gnu
```

#### Usage
Inference build will generate spectre_vit binary. It supports the model path argument. In order to run inference with a specified model, use the following command:
```
./spectre_vit model_name.onnx
```
It will look for model_name.onnx and model.onnx.data in the specified path.
At this moment, ort api doesn't support custom names for data files. Thus, you're required to name data file 'model.onnx.data' exactly. 
