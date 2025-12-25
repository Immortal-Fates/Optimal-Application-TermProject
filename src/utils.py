from __future__ import annotations

import logging
import os
import random
from contextlib import nullcontext
from datetime import datetime
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_run_id(prefix: str) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_logging(rank: int, log_dir: Optional[str]) -> logging.Logger:
    logger = logging.getLogger()
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    handlers: list[logging.Handler] = []
    if rank == 0:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        handlers.append(stream)
    if log_dir:
        ensure_dir(log_dir)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"rank{rank}.log"))
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    for handler in handlers:
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def resolve_precision(precision: str) -> str:
    precision = precision.lower()
    if precision == "auto":
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"Unsupported precision: {precision}")
    return precision


def get_autocast_context(precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.cuda.amp.autocast(dtype=dtype)


def maybe_create_grad_scaler(precision: str) -> Optional[torch.cuda.amp.GradScaler]:
    if precision == "fp16":
        return torch.cuda.amp.GradScaler()
    return None
