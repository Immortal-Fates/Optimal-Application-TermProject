# Optimal Control Term Project: Dynamic Micro-Batch Scheduling for DDP

Project site: https://immortal-fates.github.io/Optimal-Application-TermProject/

This repo studies per-rank imbalance in distributed training with variable-length NLP inputs and a dynamic per-GPU micro-batch scheduler that reduces tail latency and improves balance.

## Highlights

- Static vs dynamic micro-batching modes for controlled comparisons.
- Per-step/per-rank metrics collection and post-run analysis.
- Report artifacts and figures in `docs/report`.

## Quick Start

```bash
pip install -r requirements.txt
```

```bash
# Run the full pipeline (static + dynamic + analysis)
bash scripts/run_all.sh
```

## Reproduce a Run

```bash
# Static baseline
torchrun --nproc_per_node=4 -m src.train_ddp \
  --mode static --dataset sst2 --steps 300 --global_batch 32 --max_length 256

# Dynamic scheduler
torchrun --nproc_per_node=4 -m src.train_ddp \
  --mode dynamic --dataset sst2 --steps 300 --global_batch 32 --max_length 256

# Analyze results
python -m analysis.analyze_metrics \
  --run_dir outputs/<run_id_static> \
  --compare_dir outputs/<run_id_dynamic>
```

## Repository Layout

```
.
├─ analysis/         analysis scripts and plots
├─ datasets/         dataset cache (SST-2)
├─ docs/             GitHub Pages site + report assets
├─ outputs/          run artifacts and metrics
├─ scripts/          entrypoint shell scripts
├─ src/              training, scheduler, and utils
├─ requirements.txt
└─ README.md
```

## Notes

- The training code currently supports SST-2 only (`--dataset sst2`).
- Results land in `outputs/<run_id>/` with merged metrics and figures.
