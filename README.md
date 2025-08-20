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
python main.py --cfg configs/GT/2_MPNN/GatedGCN/zinc/zinc-GatedGCN-noPE.yaml accelerator "cuda:0" seed 0 dataset.dir './datasets'
# replace 'cuda:0' with the device to use
# replace 'xx/xx/data' with your data-dir (by default './datasets")
# replace 'configs/GRIT/zinc-GRIT.yaml' with any experiments to run
# configs/GT/2_MPNN/GatedGCN/zinc/zinc-GatedGCN-noPE.yaml

# configs/GT/0_bench/GRIT/cifar10/cifar10-GRIT-noPE.yaml
# configs/GT/0_bench/GRIT/cluster/cluster-GRIT-noPE.yaml
# configs/GT/0_bench/GRIT/LRGB/peptides_func/peptides-func-GRIT-noPE.yaml
# configs/GT/0_bench/GRIT/LRGB/peptides_struct/peptides-struct-GRIT-noPE.yaml
# configs/GT/0_bench/GRIT/mnist/mnist-GRIT-noPE.yaml
# configs/GT/0_bench/GRIT/pattern/pattern-GRIT-noPE.yaml
# configs/GT/0_bench/GRIT/zinc/zinc-GRIT-noPE.yaml

# configs/GT/0_bench/GRIT/cifar10/cifar10-GRIT-RRWP.yaml
# configs/GT/0_bench/GRIT/cluster/cluster-GRIT-RRWP.yaml
# configs/GT/0_bench/GRIT/LRGB/peptides_func/peptides-func-GRIT-RRWP.yaml
# configs/GT/0_bench/GRIT/LRGB/peptides_struct/peptides-struct-GRIT-RRWP.yaml
# configs/GT/0_bench/GRIT/mnist/mnist-GRIT-RRWP.yaml
# configs/GT/0_bench/GRIT/pattern/pattern-GRIT-RRWP.yaml
# configs/GT/0_bench/GRIT/zinc/zinc-GRIT-RRWP.yaml

# GatedGCN
# GIN
# GINE

# configs/GT/2_MPNN/GINE/cluster/cluster-GINE-noPE.yaml

# configs/GT/2_MPNN/GatedGCN/cifar10/cifar10-GatedGCN-noPE.yaml
# configs/GT/2_MPNN/GatedGCN/cluster/cluster-GatedGCN-noPE.yaml

# configs/GT/2_MPNN/GatedGCN/LRGB/COCO/coco-GatedGCN-noPE.yaml
# configs/GT/2_MPNN/GatedGCN/LRGB/pcqm_contact/pcqm-contact-GatedGCN-noPE.yaml
# configs/GT/2_MPNN/GatedGCN/LRGB/peptides_func/peptides-func-GatedGCN-noPE.yaml
# configs/GT/2_MPNN/GatedGCN/LRGB/peptides_struct/peptides-struct-GatedGCN-noPE.yaml
# configs/GT/2_MPNN/GatedGCN/LRGB/VOC/voc-GatedGCN-noPE.yaml

# configs/GT/2_MPNN/GatedGCN/mnist/mnist-GatedGCN-noPE.yaml
# configs/GT/2_MPNN/GatedGCN/pattern/pattern-GatedGCN-noPE.yaml
# configs/GT/2_MPNN/GatedGCN/zinc/zinc-GatedGCN-noPE.yaml

# configs/GT/2_MPNN/GatedGCN/cifar10/cifar10-GatedGCN-RRWP.yaml
# configs/GT/2_MPNN/GatedGCN/mnist/mnist-GatedGCN-RRWP.yaml
# configs/GT/2_MPNN/GatedGCN/zinc/zinc-GatedGCN-RWSE.yaml
# configs/GT/2_MPNN/GatedGCN/pattern/pattern-GatedGCN-RWSE.yaml
# configs/GT/2_MPNN/GatedGCN/cluster/cluster-GatedGCN-SignNet.yaml

# configs/GT/2_MPNN/GatedGCN/synthetic/fourcycles/fourcycles-GatedGCN-noPE.yaml
# configs/GT/2_MPNN/GatedGCN/synthetic/limitsone/limitsone-GatedGCN-noPE.yaml
# configs/GT/2_MPNN/GatedGCN/synthetic/limitstwo/limitstwo-GatedGCN-noPE.yaml
# configs/GT/2_MPNN/GatedGCN/synthetic/skipcircles/skipcircles-GatedGCN-noPE.yaml
# configs/GT/2_MPNN/GatedGCN/synthetic/triangles/triangles-GatedGCN-noPE.yaml

```

### Configurations and Scripts

- Configurations are available under `PEGT/configs/GT/0_bench/xx/dataset/dataset-xx-yy.yaml` where
dataset is the name of the dataset, xx is the attention module and yy is your positional encoding
- Scripts to execute are available under `./scripts/xxx.sh`
  - will run 4 trials of experiments parallelly on `GPU:0,1,2,3`. 
