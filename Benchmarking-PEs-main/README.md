# PEGT: Evaluating Positional Encodings for Graph Transformers
This repo is the extension of GRIT to evaluate PE on GTs.


> The implementation is based on [GRIT (Ma et al., ICML 2023)](https://jiaqingxie.github.io/paper/GRIT.pdf).
 

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

### Running PEGT
```bash
# Run
python main.py --cfg configs/GT/2_MPNN/GatedGCN/zinc/zinc-GatedGCN-noPE.yaml accelerator "cuda:0" seed 0 dataset.dir '../datasets'
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

# configs/GT/2_MPNN/GIN/cifar10/cifar10-GIN-noPE.yaml
```

### Implemented Graph Transformers with Sparse Attention
- Exphormer (included) `grit/layer/Exphormer.py`
- GraphGPS (included) `grit/layer/gps_layer.py`
- NodeFormer (included) `grit/layer/nodeformer_layer.py`
- DIFFORMER (included) `grit/layer/difformer_layer.py`
- GOAT (included) `grit/layer/goat_layer.py`
- NAGphormer (included, adapted to graph level) `grit/layer/nagphormer_layer.py`

### Implemented Graph Transformers with Global Attention
- GRIT (included) `grit/layer/grit_layer.py`
- Graphormer (included) `grit/layer/graphormer_layer.py`
- EGT (included) `grit/layer/egt_layer.py`
- SAN (included) `grit/layer/san_layer.py`
- GraphTrans (included) `grit/layer/graphtrans_layer.py`
- GraphiT (included) `grit/layer/graphit_layer.py`
- Original_GT (included) `grit/layer/origin_gt_layer.py`
- UniMP (included) `grit/layer/unimp.py`
- SAT (included) `grit/layer/SAT_layer.py`



### Implemented Existing Positional Encoding in Graph Transformers (before April 2024)

- ESLapPE (Already in GPS/GRIT) `grit/encoder/equivstable_laplace_pos_encoder.py`
- LapPE (Already in GPS/GRIT) `grit/encoder/laplace_pos_encoder.py`
- RWSE  (Already in GPS/GRIT) `grit/encoder/kernel_pos_encoder.py`
- RRWP  (Already in GRIT) `grit/encoder/rrwp_encoder.py`
- SPD (Already in GRIT) `grit/encoder/spd_encoder.py`
- SignNet (Already in GPS/GRIT)  `grit/encoder/signnet_pos_encoder.py`
- Personalized Page Rank (PPR) `grit/encoder/ppr_pos_encoder.py`
- SVD-based PE (SVD) `grit/encoder/svd_pos_encoder.py`
- Node2Vec Algorithm (NODE2VEC) `grit/encoder/node2vec_pos_encoder.py`
- WL test based PE (WLPE) `grit/encoder/wlpe_pos_encoder.py`
- Diffusion on Kernelized Laplacian PE (GCKN) `grit/encoder/gckn_pos_encoder.py`
- Diffusion on Random Walk Probabilities (LSPE) `grit/encoder/rwdiff_pos_encoder.py`
- CORE Graph Rewiring and Drawing (CORE) `grit/encoder/gd_encoder.py`

### Configurations and Scripts

- Configurations are available under `PEGT/configs/GT/0_bench/xx/dataset/dataset-xx-yy.yaml` where
dataset is the name of the dataset, xx is the attention module and yy is your positional encoding
- Scripts to execute are available under `./scripts/xxx.sh`
  - will run 4 trials of experiments parallelly on `GPU:0,1,2,3`. 
