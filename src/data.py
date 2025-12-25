from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Generator, Iterable, List

from datasets import load_dataset
from transformers import AutoTokenizer


class SampleBuffer:
    """FIFO sample buffer used for per-step prefetch and scheduling."""
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buffer: Deque[Dict] = deque()

    def __len__(self) -> int:
        return len(self._buffer)

    def fill(self, iterator: Iterable[Dict]) -> None:
        while len(self._buffer) < self.capacity:
            self._buffer.append(next(iterator))

    def pop(self, n: int) -> List[Dict]:
        if n > len(self._buffer):
            raise ValueError(f"Buffer underflow: need {n}, have {len(self._buffer)}")
        batch: List[Dict] = []
        for _ in range(n):
            batch.append(self._buffer.popleft())
        return batch

    def peek_lengths(self, max_k: int) -> List[int]:
        lengths: List[int] = []
        for idx, sample in enumerate(self._buffer):
            if idx >= max_k:
                break
            lengths.append(int(sample["length"]))
        return lengths


def load_sst2(
    tokenizer_name: str,
    split: str,
    max_length: int,
    seed: int,
    cache_dir: str,
    max_samples: int | None = None,
):
    """Load SST-2, tokenize, and attach a per-sample length field."""
    raw = load_dataset("glue", "sst2", split=split, cache_dir=cache_dir)
    if max_samples is not None:
        raw = raw.shuffle(seed=seed).select(range(max_samples))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def tokenize_batch(batch):
        tokens = tokenizer(batch["sentence"], truncation=True, max_length=max_length)
        tokens["labels"] = batch["label"]
        tokens["length"] = [len(ids) for ids in tokens["input_ids"]]
        return tokens

    dataset = raw.map(tokenize_batch, batched=True, remove_columns=raw.column_names)
    dataset.set_format(type="python")
    return dataset, tokenizer


def infinite_sharded_iterator(
    dataset,
    rank: int,
    world_size: int,
    seed: int,
) -> Generator[Dict, None, None]:
    """Yield an infinite stream of shuffled, rank-sharded samples."""
    epoch = 0
    while True:
        shuffled = dataset.shuffle(seed=seed + epoch)
        shard = shuffled.shard(num_shards=world_size, index=rank, contiguous=True)
        for item in shard:
            yield item
        epoch += 1
