# Dynamic Per-GPU Micro-Batch Scheduling for DDP (SST-2)

This project provides a fast, reproducible DDP experiment that demonstrates per-rank imbalance caused by variable-length NLP inputs, and a dynamic per-GPU micro-batch scheduler that improves step-time tail latency and resource balance.

Key features:
- `static` baseline: fixed micro-batch per GPU
- `dynamic`: per-step min-max integer optimization using a length-based cost proxy
- per-step/per-rank metrics recording and merged analysis
- quick default run that still produces visible differences and plots

## Install

```bash
pip install -r requirements.txt
```

## Quick Run

```bash
# 1) baseline
torchrun --nproc_per_node=4 -m src.train_ddp \
  --mode static --dataset sst2 --steps 300 --global_batch 32 --max_length 256

# 2) dynamic
torchrun --nproc_per_node=4 -m src.train_ddp \
  --mode dynamic --dataset sst2 --steps 300 --global_batch 32 --max_length 256

# 3) analysis
python -m analysis.analyze_metrics \
  --run_dir outputs/<run_id_static> \
  --compare_dir outputs/<run_id_dynamic>
```

Or run everything:

```bash
bash scripts/run_all.sh
```

## Layout

```
.
├─ README.md
├─ requirements.txt
├─ scripts/
│  ├─ run_all.sh
│  ├─ run_static.sh
│  └─ run_dynamic.sh
├─ src/
│  ├─ train_ddp.py
│  ├─ scheduler.py
│  ├─ data.py
│  ├─ metrics.py
│  ├─ distributed.py
│  └─ utils.py
├─ analysis/
│  └─ analyze_metrics.py
├─ report/
│  └─ report.md
└─ outputs/
   └─ <run_id>/
      ├─ config.yaml
      ├─ metrics/
      │  ├─ rank0.csv
      │  ├─ rank1.csv
      │  ├─ rank2.csv
      │  └─ rank3.csv
      ├─ merged_metrics.csv
      ├─ figures/
      │  └─ *.png
      └─ summary.csv
```

## Outputs

Each run creates a unique `run_id` directory:
- `metrics/rank*.csv`: raw per-step/per-rank metrics
- `merged_metrics.csv`: rank0 merged table
- `figures/*.png`: analysis plots
- `summary.csv`: summary stats for static vs dynamic

The latest run directory is recorded in:
- `outputs/last_run_static.txt`
- `outputs/last_run_dynamic.txt`

## Implementation Notes

- Dynamic scheduler: prefetch K samples, compute prefix `max_len[k]` and `padded_tokens[k]`, then solve a min-max integer program per step.
- DDP gradient scaling: apply `L_g * (G * b_g / B)` so allreduce gives the global mean gradient even with per-rank batch differences.
- OOM protection: truncation to `max_length`, plus startup probing to find `b_max_g` on each GPU.

## Example Flags

```bash
torchrun --nproc_per_node=4 -m src.train_ddp \
  --mode dynamic \
  --steps 200 \
  --global_batch 32 \
  --buffer_size 32 \
  --cost_type tokens_sq \
  --precision auto \
  --max_length 256 \
  --dataset_dir datasets
```

## Report

The report skeleton is in `report/report.md`. Insert figures from `analysis` and summarize the results.
