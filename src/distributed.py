from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize process group and return local_rank, rank, world_size."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def get_rank() -> int:
    """Return current rank or 0 when not initialized."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Return world size or 1 when not initialized."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    """True when rank is 0."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all ranks."""
    if dist.is_initialized():
        dist.barrier()
