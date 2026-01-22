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