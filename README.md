
## Installation and Requirements

Please note that our environment requirements are different from LLaVA's environment requirements. We strongly recommend you create the environment from scratch as follows.

1. Clone this repository and navigate to the folder
```bash
git clone []
cd src_biomolecule_insruction_tuning
```

2. Create a conda environment, activate it and install Packages
```Shell
conda create -n kronos_instr python=3.10 -y
conda activate kronos_instr
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages
```Shell
pip install flash-attn==2.5.7 --no-build-isolation
pip install torch-geometric==2.3.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```
#### Upgrade to the latest code base

```Shell
git pull
pip install -e .
```

## Train
The scripts are grouped into 3 different types of models:

1. MLP Encoding for Representation of Proteomics Data
2. Patient Similarity Node Encoding for Representation of Proteomics Data
3. Biomolecular Context Injection: Graph Encoding for Representation of Proteomics Data (KRONOS)

```Shell
conda activate kronos_instr
bash DEEPSPEED_[pretrain/finetune]_[mlp/node/graph]_proteomics.sh
```