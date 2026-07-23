# KiteNorm: Variance Regularisation for Stable and Scalable Post-LN Transformers

This respository serves as a reference implementation based on PlainLM: https://github.com/Niccolo-Ajroldi/plainLM

arXiv: (Coming soon)

Instructions are repeated below.

## Installation
We recommend running `plainLM` in a dedicated Python environment. To install dependencies in an Anaconda environment, execute:
```bash
conda create --name plainLM python=3.12 -y && conda activate plainLM
pip install -r requirements.txt
```

## 📚 Data
We provide a script for downloading, tokenizing, chunking and saving Hugging Face datasets: `data/datasets/slim_pajama/prepare_train.py`.
You can specify any HF dataset and tokenizer. To avoid downloading the entire corpus, we stream, tokenize, and chunk data on-the-fly.

## ⚡️ Usage

Specify hyperparameters in `config.yaml` and launch training as follows:

#### Single GPU/CPU:
```bash
  python train.py --config=config/config.yaml
```
#### Multiple GPUs:
```bash
  torchrun --nnodes=1 --nproc_per_node=4 train.py --config=code/config/sweep.yaml
```
