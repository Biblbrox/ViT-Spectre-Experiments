# Spectre ViT experiment project

## Content
Important folders:
- spectre_vit -- python pytorch experiments folder;
- spectre_vit/repl -- repl scrips for training, testing, and other experiments;
- spectre_vit/configs -- a directory where all config files are stored. If you want to conduct a new experiment, create a separate config file in that directory.
- spectre_vit/models -- model class files

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

If you want to use distillation in your studies, make sure that dinov3 submodule is cloned:
```
git submodule update --init --recursive && uv sync
```

Now you should be able to choose spectre_vit environment in every jupyter provided notebook.

### Notebooks
- vit_spectre_cifar100.ipynb - Cifar100 training pipeline;
- vit_spectre_dino_pretraining.ipynb - DINOv3 fine-tuning pipeline;
- vit_spectre_mnist.ipynb - MNIST100 traning pipeline;
- fft_experiments.ipynb - Visualization of fft features from the model;
- dwt_experiments.ipynb - Visualization of dwt features from the model.
