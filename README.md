# Alpha-Asynchronous Updates in Graph Neural Networks
This repo is based on [Benchmarking-PEs (Grötschla et al., 2024)](https://github.com/ETH-DISCO/Benchmarking-PEs).
 

### Python environment setup with Conda
```bash
conda create -n grit python=3.9
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
conda install pyg -c pyg
conda install openbabel fsspec rdkit -c conda-forge
pip install yacs torchmetrics
pip install performer-pytorch
pip install ogb
pip install tensorboardX
pip install wandb
pip install torch_ppr
pip install attrdict
pip install opt_einsum
pip install graphgym
pip install setuptools==59.5.0
pip install loguru
pip install pytorch_lightning
conda clean --all

```

### Running an experiment
```bash
# Run
python main.py --cfg configs/GT/2_MPNN/GatedGCN/zinc/zinc-GatedGCN-noPE.yaml accelerator "cuda:0" dataset.dir 'xx/xx/data' async_update.alpha "a"
# replace 'cuda:0' with the device to use
# replace 'xx/xx/data' with your data-dir (by default './datasets")
# replace 'configs/GT/2_MPNN/GatedGCN/zinc/zinc-GatedGCN-noPE.yaml' with any experiments to run
# replace 'a' with the chosen alpha ([0,1]) (probability of a node to perform an update, default: 1)

```

### Configurations

- Configurations are available under `configs/GT/2_MPNN/GatedGCN/dataset/dataset-GatedGCN-yy.yaml` where
dataset is the name of the dataset and yy is your positional encoding
  - The possible options for asynchronous message passing grouped under "async_update" in each config
