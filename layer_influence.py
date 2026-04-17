"""Measure validation loss when pruning model layers.

Logs to WandB a TSV table with the columns:

    removed_layer_idx    valid_loss

The baseline model loss is reported with ``removed_layer_idx = -1``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
from pathlib import Path

import torch
import wandb
from datasets import Dataset, load_from_disk
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler

import utils
from models import construct_model


def _resolve_layer_scaled(v, n_layers: int):
    if isinstance(v, str):
        s = v.strip().lower().replace(" ", "")
        if s == "2l":
            return float(2 * n_layers)
        if s == "1/sqrt(l)":
            return 1 / math.sqrt(n_layers)
        if s == "1/l":
            return float(1.0 / n_layers)
        if s == "1/(2l)":
            return 1.0 / float(2 * n_layers)
    return v


def _prepare_cfg(cfg):
    skip_scale = getattr(cfg, "skip_scale", -1)
    res_scale = getattr(cfg, "res_scale", -1)
    cfg = cfg._replace(
        skip_scale=skip_scale if skip_scale == -1 else _resolve_layer_scaled(skip_scale, cfg.n_layers),
        res_scale=res_scale if res_scale == -1 else _resolve_layer_scaled(res_scale, cfg.n_layers),
    )
    if hasattr(cfg, "torch_compile"):
        cfg = cfg._replace(torch_compile=False)
    return cfg


def _select_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autocast_context(device: str, dtype_name: str):
    if "cuda" not in device:
        return contextlib.nullcontext()
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]
    return torch.amp.autocast(device_type="cuda", dtype=ptdtype)


def _make_validloader(cfg, device: str) -> DataLoader:
    if not cfg.validset_path:
        raise ValueError("Config must define validset_path for layer influence evaluation.")

    valid_set = load_from_disk(cfg.validset_path)
    if not isinstance(valid_set, Dataset):
        raise ValueError("Validation dataset should be a datasets.Dataset.")

    return DataLoader(
        valid_set,
        batch_size=cfg.micro_batch_size,
        num_workers=cfg.num_workers,
        sampler=SequentialSampler(valid_set),
        pin_memory=("cuda" in device),
    )


def _move_to_device(batch, seq_len: int, device: str):
    inputs = batch["input_ids"][:, :seq_len]
    targets = batch["input_ids"][:, 1 : (seq_len + 1)]

    if "cuda" in device:
        inputs = inputs.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
    else:
        inputs = inputs.to(device)
        targets = targets.to(device)

    return inputs, targets


@torch.no_grad()
def _eval_valid_loss(
    model,
    dataloader,
    seq_len: int,
    device: str,
    dtype_name: str,
    max_eval_batches: int | None,
) -> float:
    criterion = CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0

    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break

        inputs, targets = _move_to_device(batch, seq_len, device)
        with _autocast_context(device, dtype_name):
            output = model(inputs)
            logits = getattr(output, "logits", output)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        if loss is None or torch.isnan(loss):
            raise ValueError("Validation loss is nan.")

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("No validation batches were evaluated.")

    return total_loss / num_batches


def _is_ignorable_collector_key(key: str) -> bool:
    return ".coll_" in key and (
        key.endswith(".running_var_sum")
        or key.endswith(".running_count_1")
        or key.endswith(".running_kurtosis_sum")
        or key.endswith(".running_count_2")
        or key.endswith(".running_token_cos_alignment_sum")
        or key.endswith(".running_token_non_mean_portion_sum")
        or key.endswith(".running_count_3")
    )


def _load_model(cfg, checkpoint_path: str, device: str):
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = construct_model(cfg)

    if not hasattr(model, "layers"):
        raise ValueError("Model does not expose a `layers` attribute.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing `state_dict`.")

    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    bad_missing = [k for k in incompatible.missing_keys if not _is_ignorable_collector_key(k)]
    bad_unexpected = [k for k in incompatible.unexpected_keys if not _is_ignorable_collector_key(k)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with model.\n"
            f"Missing keys: {bad_missing}\n"
            f"Unexpected keys: {bad_unexpected}"
        )

    model.to(device)
    model.eval()
    return model


def _ablate_layer_literal(model, layer_idx: int) -> None:
    if not (0 <= layer_idx < len(model.layers)):
        raise IndexError(f"Layer index out of range: {layer_idx}")
    del model.layers[layer_idx]


def _format_rows(rows: list[tuple[int, float]]) -> str:
    lines = ["removed_layer_idx\tvalid_loss"]
    for removed_layer_idx, valid_loss in rows:
        lines.append(f"{removed_layer_idx}\t{valid_loss:.10f}")
    return "\n".join(lines) + "\n"


def _wandb_run_name(cfg) -> str:
    return f"LI_{cfg.wandb_run_name}_{cfg.lr}"


def _artifact_name(run_name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name) + "_layer_influence"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="Path to a checkpoint created by train.py.")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Optional cap on validation batches.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--wandb_project", required=True, help="Weights & Biases project name.")
    parser.add_argument("--wandb_dir", required=True, help="Weights & Biases output directory.")
    args = parser.parse_args()

    run = wandb.init(
        project=args.wandb_project,
        dir=args.wandb_dir,
    )

    try:
        checkpoint_path = Path(args.checkpoint_path)
        config_path = checkpoint_path.parent / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Expected config next to checkpoint: {config_path}")

        cfg, _ = utils.load_config(str(config_path))
        cfg = _prepare_cfg(cfg)
        run.name = _wandb_run_name(cfg)
        run.config.update(cfg._asdict())

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = getattr(cfg, "cuda_matmul_allow_tf32", False)
            torch.backends.cudnn.allow_tf32 = getattr(cfg, "cudnn_allow_tf32", True)

        device = _select_device(args.device)
        validloader = _make_validloader(cfg, device)

        baseline_model = _load_model(cfg, str(checkpoint_path), device)
        n_layers = len(baseline_model.layers)
        rows = [(-1, _eval_valid_loss(baseline_model, validloader, cfg.seq_len, device, cfg.dtype, args.max_eval_batches))]

        del baseline_model
        if "cuda" in device:
            torch.cuda.empty_cache()

        for layer_idx in range(n_layers):
            model = _load_model(cfg, str(checkpoint_path), device)
            _ablate_layer_literal(model, layer_idx)
            valid_loss = _eval_valid_loss(model, validloader, cfg.seq_len, device, cfg.dtype, args.max_eval_batches)
            rows.append((layer_idx, valid_loss))
            del model
            if "cuda" in device:
                torch.cuda.empty_cache()

        table = _format_rows(rows)
        artifact = wandb.Artifact(_artifact_name(run.name), type="layer_influence")
        with artifact.new_file("layer_influence.tsv", mode="w", encoding="utf-8") as f:
            f.write(table)
        run.log_artifact(artifact)

        print(table, end="")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
