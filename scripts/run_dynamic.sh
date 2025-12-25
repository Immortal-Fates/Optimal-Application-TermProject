#!/usr/bin/env bash
set -euo pipefail

torchrun --nproc_per_node=4 -m src.train_ddp \
  --mode dynamic \
  --dataset sst2 \
  --steps 800 \
  --global_batch 64 \
  --max_length 256 \
  --buffer_size 64 \
  --cost_type tokens
