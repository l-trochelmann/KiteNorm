"""Measure downstream task scores when pruning model layers.

Logs to WandB a TSV table with one row per model variant. The baseline model is
reported with ``removed_layer_idx = -1``.

Default tasks are small-model friendly lm-eval tasks:

    piqa,arc_easy,winogrande
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import wandb
from transformers import AutoTokenizer

import utils
from layer_influence import (
    _ablate_layer_literal,
    _autocast_context,
    _load_model,
    _prepare_cfg,
    _select_device,
)


DEFAULT_TASKS = "piqa,arc_easy,winogrande"
DEFAULT_HF_CACHE_DIR = "/fast/ltrochelmann/data/lm/lm-eval"
DEFAULT_PRIMARY_METRICS = (
    "piqa/acc",
    "piqa/acc_norm",
    "arc_easy/acc",
    "arc_easy/acc_norm",
    "winogrande/acc",
)


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_limit(value: str | None) -> int | float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"none", "null"}:
        return None
    if "." in stripped:
        return float(stripped)
    return int(stripped)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_tokenizer_path(cfg, tokenizer_path: str | None) -> str:
    if tokenizer_path:
        return tokenizer_path

    cfg_tokenizer_path = getattr(cfg, "tokenizer_path", None)
    if cfg_tokenizer_path:
        return cfg_tokenizer_path

    trainset_path = getattr(cfg, "trainset_path", None)
    if trainset_path:
        candidate = Path(trainset_path).parent / "tokenizer"
        if candidate.is_dir():
            return str(candidate)

    validset_path = getattr(cfg, "validset_path", None)
    if validset_path:
        candidate = Path(validset_path).parent / "tokenizer"
        if candidate.is_dir():
            return str(candidate)

    if getattr(cfg, "vocab_size", None) == 50257:
        return "gpt2"
    return "EleutherAI/gpt-neox-20b"


def _load_tokenizer(cfg, tokenizer_path: str | None):
    resolved = _resolve_tokenizer_path(cfg, tokenizer_path)
    _log(f"Loading tokenizer from: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, resolved


def _configure_hf_cache(cache_dir: str | None) -> str | None:
    if cache_dir is None:
        return None

    root = Path(cache_dir).expanduser()
    datasets_cache = root / "datasets"
    hub_cache = root / "hub"
    modules_cache = root / "modules"
    transformers_cache = root / "transformers"

    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    modules_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_MODULES_CACHE", str(modules_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))

    _log(f"Using HF cache root: {root}")
    _log(f"Using HF datasets cache: {datasets_cache}")
    _log(f"Using HF hub cache: {hub_cache}")
    _log(f"Using HF modules cache: {modules_cache}")

    return str(root)


def _request_args(request: Any):
    return request.args if hasattr(request, "args") else request


def _make_lm_eval_wrapper(
    model,
    tokenizer,
    device: str,
    dtype_name: str,
    max_length: int,
    batch_size: int,
):
    try:
        from lm_eval.api.model import LM
    except ImportError as exc:
        raise ImportError(
            "lm_eval is required for downstream layer influence evaluation. "
            "Install the pinned package in your environment, e.g. lm-eval==0.4.8."
        ) from exc

    class NativeTransformerLM(LM):
        def __init__(self):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self._device = torch.device(device)
            self._max_length = max_length
            self._batch_size = batch_size
            self._pad_token_id = tokenizer.pad_token_id
            self._eot_token_id = tokenizer.eos_token_id
            if self._pad_token_id is None:
                self._pad_token_id = self._eot_token_id
            if self._eot_token_id is None:
                self._eot_token_id = self._pad_token_id

        @property
        def eot_token_id(self):
            return self._eot_token_id

        @property
        def max_length(self):
            return self._max_length

        @property
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def device(self):
            return self._device

        def tok_encode(self, string: str, **kwargs):
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens)

        def _encode_pair(self, context: str, continuation: str):
            whole = self.tok_encode(context + continuation)
            context_enc = self.tok_encode(context)
            continuation_enc = whole[len(context_enc) :]
            if not continuation_enc:
                continuation_enc = self.tok_encode(continuation)
                whole = context_enc + continuation_enc
            return context_enc, continuation_enc, whole

        def _prepare_request(self, context: str, continuation: str):
            context_enc, continuation_enc, whole = self._encode_pair(context, continuation)
            continuation_start = len(context_enc)
            if continuation_start == 0:
                whole = [self.eot_token_id] + whole
                continuation_start = 1

            if len(whole) > self.max_length:
                left_truncation = len(whole) - self.max_length
                whole = whole[left_truncation:]
                continuation_start = max(continuation_start - left_truncation, 0)

            score_start = max(continuation_start, 1)
            score_positions = list(range(score_start, len(whole)))
            if continuation_enc and len(score_positions) > len(continuation_enc):
                score_positions = score_positions[-len(continuation_enc) :]

            return {
                "input_ids": whole,
                "score_positions": score_positions,
            }

        @torch.no_grad()
        def loglikelihood(self, requests):
            prepared = [
                self._prepare_request(*_request_args(request))
                for request in requests
            ]
            results = []

            for start in range(0, len(prepared), self.batch_size):
                batch = prepared[start : start + self.batch_size]
                batch_max_len = max(len(item["input_ids"]) for item in batch)
                input_ids = torch.full(
                    (len(batch), batch_max_len),
                    self._pad_token_id,
                    dtype=torch.long,
                    device=self.device,
                )
                for row_idx, item in enumerate(batch):
                    ids = torch.tensor(item["input_ids"], dtype=torch.long, device=self.device)
                    input_ids[row_idx, : ids.numel()] = ids

                with _autocast_context(device, dtype_name):
                    output = self.model(input_ids)
                    logits = getattr(output, "logits", output)

                log_probs = torch.log_softmax(logits.float(), dim=-1)

                for row_idx, item in enumerate(batch):
                    total = 0.0
                    greedy = True
                    ids = item["input_ids"]
                    for pos in item["score_positions"]:
                        token_id = ids[pos]
                        prev_logits = log_probs[row_idx, pos - 1]
                        total += prev_logits[token_id].item()
                        greedy = greedy and (int(torch.argmax(prev_logits).item()) == token_id)
                    results.append((total, greedy))

            return results

        def loglikelihood_rolling(self, requests):
            synthetic_requests = []
            for request in requests:
                (text,) = _request_args(request)
                synthetic_requests.append(("", text))
            return [score for score, _ in self.loglikelihood(synthetic_requests)]

        def generate_until(self, requests):
            raise NotImplementedError(
                "layer_influence_lm-eval.py currently targets loglikelihood "
                "tasks only; use piqa, arc_easy, and winogrande."
            )

    return NativeTransformerLM()


def _run_lm_eval(
    model,
    tokenizer,
    tasks: list[str],
    device: str,
    dtype_name: str,
    max_length: int,
    batch_size: int,
    limit: int | float | None,
):
    _log(
        "Starting lm-eval: "
        f"tasks={','.join(tasks)}, limit={limit}, batch_size={batch_size}"
    )
    try:
        from lm_eval import evaluator
    except ImportError as exc:
        raise ImportError(
            "lm_eval is required for downstream layer influence evaluation. "
            "Install the pinned package in your environment, e.g. lm-eval==0.4.8."
        ) from exc

    lm = _make_lm_eval_wrapper(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype_name=dtype_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    try:
        return evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=0,
            batch_size=batch_size,
            device=device,
            limit=limit,
        )
    except TypeError:
        return evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=0,
            batch_size=batch_size,
            limit=limit,
        )


def _flatten_lm_eval_results(results: dict) -> dict[str, float]:
    flattened = {}
    for task_name, task_results in results.get("results", {}).items():
        for metric_key, value in task_results.items():
            if metric_key == "alias" or "_stderr" in metric_key:
                continue
            if value is None or value == "N/A":
                continue
            metric_name = metric_key.split(",")[0]
            flattened[f"{task_name}/{metric_name}"] = float(value)
    return flattened


def _format_rows(rows: list[tuple[int, dict[str, float]]], columns: list[str]) -> str:
    lines = ["removed_layer_idx\t" + "\t".join(columns)]
    for removed_layer_idx, metrics in rows:
        values = [f"{metrics[column]:.10f}" if column in metrics else "nan" for column in columns]
        lines.append(f"{removed_layer_idx}\t" + "\t".join(values))
    return "\n".join(lines) + "\n"


def _wandb_run_name(cfg) -> str:
    return f"LI_lm-eval_{cfg.wandb_run_name}_{cfg.lr}"


def _artifact_name(run_name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name) + "_artifact"


def _artifact_tsv_filename(cfg) -> str:
    return f"{cfg.wandb_run_name}-{cfg.lr}.tsv"


def _metric_columns(rows: list[tuple[int, dict[str, float]]]) -> list[str]:
    observed = {key for _, metrics in rows for key in metrics}
    columns = [key for key in DEFAULT_PRIMARY_METRICS if key in observed]
    columns.extend(sorted(observed - set(columns)))
    return columns


def _evaluate_variant(
    cfg,
    checkpoint_path: Path,
    tokenizer,
    removed_layer_idx: int,
    device: str,
    tasks: list[str],
    batch_size: int,
    limit: int | float | None,
) -> dict[str, float]:
    label = "baseline" if removed_layer_idx < 0 else f"removed_layer_idx={removed_layer_idx}"
    _log(f"Loading model variant: {label}")
    model = _load_model(cfg, str(checkpoint_path), device)
    if removed_layer_idx >= 0:
        _log(f"Ablating layer {removed_layer_idx}")
        _ablate_layer_literal(model, removed_layer_idx)

    _log(f"Evaluating model variant: {label}")
    results = _run_lm_eval(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        device=device,
        dtype_name=cfg.dtype,
        max_length=cfg.seq_len,
        batch_size=batch_size,
        limit=limit,
    )
    metrics = _flatten_lm_eval_results(results)
    _log(f"Finished model variant: {label} -> {metrics}")

    del model
    if "cuda" in device:
        torch.cuda.empty_cache()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="Path to a checkpoint created by train.py.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--wandb_project", required=True, help="Weights & Biases project name.")
    parser.add_argument("--wandb_dir", required=True, help="Weights & Biases output directory.")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Deprecated compatibility alias for --limit.")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help=f"Comma-separated lm-eval tasks. Default: {DEFAULT_TASKS}.")
    parser.add_argument("--batch_size", type=int, default=8, help="lm-eval request batch size.")
    parser.add_argument("--limit", default=None, help="Optional lm-eval limit, either an integer count or a float fraction.")
    parser.add_argument("--tokenizer_path", default=None, help="Optional tokenizer path/name. Defaults to <trainset parent>/tokenizer when present.")
    parser.add_argument("--hf_cache_dir", default=DEFAULT_HF_CACHE_DIR, help="HF/lm-eval cache root for task and tokenizer downloads.")
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
        hf_cache_dir = _configure_hf_cache(args.hf_cache_dir)
        run.name = _wandb_run_name(cfg)
        run.config.update(cfg._asdict())
        run.config.update(
            {
                "lm_eval_tasks": args.tasks,
                "lm_eval_limit": args.limit if args.limit is not None else args.max_eval_batches,
                "lm_eval_batch_size": args.batch_size,
                "hf_cache_dir": hf_cache_dir,
            },
            allow_val_change=True,
        )

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = getattr(cfg, "cuda_matmul_allow_tf32", False)
            torch.backends.cudnn.allow_tf32 = getattr(cfg, "cudnn_allow_tf32", True)

        device = _select_device(args.device)
        tasks = _split_csv(args.tasks)
        limit_arg = args.limit if args.limit is not None else (
            str(args.max_eval_batches) if args.max_eval_batches is not None else None
        )
        limit = _parse_limit(limit_arg)
        tokenizer, resolved_tokenizer = _load_tokenizer(cfg, args.tokenizer_path)
        run.config.update({"tokenizer_path_resolved": resolved_tokenizer}, allow_val_change=True)

        _log(f"Loading baseline model once to determine layer count from: {checkpoint_path}")
        baseline_model = _load_model(cfg, str(checkpoint_path), device)
        n_layers = len(baseline_model.layers)
        _log(f"Found {n_layers} layers; evaluating {n_layers + 1} model variants")
        del baseline_model
        if "cuda" in device:
            torch.cuda.empty_cache()

        rows = [
            (
                -1,
                _evaluate_variant(
                    cfg,
                    checkpoint_path,
                    tokenizer,
                    -1,
                    device,
                    tasks,
                    args.batch_size,
                    limit,
                ),
            )
        ]

        for layer_idx in range(n_layers):
            rows.append(
                (
                    layer_idx,
                    _evaluate_variant(
                        cfg,
                        checkpoint_path,
                        tokenizer,
                        layer_idx,
                        device,
                        tasks,
                        args.batch_size,
                        limit,
                    ),
                )
            )

        table = _format_rows(rows, _metric_columns(rows))
        artifact = wandb.Artifact(_artifact_name(run.name), type="layer_influence")
        with artifact.new_file(_artifact_tsv_filename(cfg), mode="w", encoding="utf-8") as f:
            f.write(table)
        run.log_artifact(artifact)

        print(table, end="")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
