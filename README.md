# KiteNorm: Variance Regularisation for Stable and Scalable Post-LN Transformers

Reference implementation for KiteNorm, based on [PlainLM](https://github.com/Niccolo-Ajroldi/plainLM).

Paper: coming soon.

## Installation

This project uses Python 3.12, PyTorch 2.7.1, and CUDA 12.8.
```bash
conda create --name kitenorm python=3.12 -y
conda activate kitenorm
python -m pip install -r requirements.txt
```

Authenticate with `wandb login`, or set `WANDB_MODE=offline`. Before use, replace path placeholders in:

- `config/TR_4xB200_KiteNorm.yaml`
- `data/datasets/slim_pajama/prepare_train.py`
- `data/datasets/slim_pajama/prepare_valid.py`
- `data/datasets/fineweb/prepare.py`
- `downstream_eval.py`

If using HTCondor, also check every `.sh` and `.sub` file under `cluster/single_gpu/` and `cluster/multi_gpu/`. Replace the example paths in the commands below as well.

## Data

Edit `dataset_name`, `out_path`, `nrows`, and `num_proc` as needed in:
- `data/datasets/slim_pajama/prepare_train.py`
- `data/datasets/slim_pajama/prepare_valid.py`
- `data/datasets/fineweb/prepare.py`

Keep `seq_len = 2048` to match the base configuration. Prepare either SlimPajama:

```bash
python data/datasets/slim_pajama/prepare_train.py
python data/datasets/slim_pajama/prepare_valid.py
```

Or FineWeb:

```bash
python data/datasets/fineweb/prepare.py
```

SlimPajama saves `train`, `validation`, and `tokenizer`. FineWeb saves `train`, `valid`, and `tokenizer`. Set the config paths accordingly.

### Reproducibility
The original `cerebras/SlimPajama-627B` dataset is no longer available on Hugging Face. Initial experiments used the original dataset, while some later experiments used a reupload. Differences in data ordering may affect exact numerical reproducibility but do not invalidate the reported observations.

One ablation using the original dataset exhibited sporadic hidden-state variance expansion and vanishing gradients, which KiteNorm prevented. This instability may not occur in a rerun using the reupload. The original data preparation used `nrows = 15_000_000` and `seq_len = 2048`.

## Configuration

Edit `config/TR_4xB200_KiteNorm.yaml` and set:

```yaml
trainset_path: '<training out_path>/train'
validset_path: '<validation out_path>/validation'
wandb_project: '<project>'
wandb_dir: '<wandb directory>'
wandb_run_name: '<run name>'
out_dir: '<checkpoint directory>'
exp_name: '<experiment name>'
```

The base configuration is a four-GPU run with learning rates `[2e-3, 5e-3]`. Select one with `--job_idx=0` or `--job_idx=1`.

## Training
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  train.py --config=config/TR_4xB200_KiteNorm.yaml --job_idx=0
```

## Downstream evaluation
```bash
python downstream_eval.py \
  --checkpoint_path="/path/to/ckpt_micro_step_N.pth" \
  --tokenizer_path="/path/to/tokenizer" \
  --hf_cache_dir="/path/to/lm-eval-cache" \
  --wandb_project="PROJECT" \
  --wandb_dir="/path/to/wandb"
```

This evaluates ARC-Easy and PIQA.
