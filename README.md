
# DDiSSAN

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

DDiSSAN:Dynamic Differential Syntax-Semantic
Attention Network for Pseudo-hidden Sentiment
Analysis

## Installation

#### Conda

```bash
# clone project
git clone https://github.com/yunsicjh/DDiSSAN.git
cd DDiSSAN

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run
We use [LAL-Parser](https://github.com/KhalilMrini/LAL-Parser) in all our
experiments to obtain the syntactic structure of
sentences. However, the datasets in this repository were obtained using the Stanford parser, so you need to reprocess the datasets with LAL-Parser.

```bash
# 进行数据处理
cd DDiSSAN/scripts
sh build_vacab.sh

# train on GPU
cd DDiSSAN/scripts
sh run.sh
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
