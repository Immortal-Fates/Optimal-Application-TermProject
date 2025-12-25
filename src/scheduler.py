from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class ScheduleResult:
    batches: List[int]
    max_cost: float
    cost_var: float
    effective_batch: int


def compute_costs(lengths: Sequence[int], max_k: int, cost_type: str) -> List[float]:
    """Compute prefix costs for k=1..max_k based on max length and padding proxy."""
    if max_k <= 0:
        return []
    max_len = 0
    costs: List[float] = []
    for idx in range(max_k):
        max_len = max(max_len, int(lengths[idx]))
        padded_tokens = (idx + 1) * max_len
        if cost_type == "tokens":
            cost = float(padded_tokens)
        elif cost_type == "tokens_sq":
            cost = float(padded_tokens**2)
        else:
            raise ValueError(f"Unsupported cost_type: {cost_type}")
        costs.append(cost)
    return costs


class DynamicBatchScheduler:
    """Exact min-max scheduler using enumeration for small world_size."""
    def __init__(self, world_size: int, cost_type: str) -> None:
        self.world_size = world_size
        self.cost_type = cost_type

    def solve(
        self,
        costs_per_rank: Sequence[Sequence[float]],
        b_max: Sequence[int],
        target_batch: int,
    ) -> ScheduleResult:
        """Solve the per-step min-max integer allocation with variance tie-break."""
        if len(costs_per_rank) != self.world_size:
            raise ValueError("costs_per_rank length mismatch with world_size")
        if len(b_max) != self.world_size:
            raise ValueError("b_max length mismatch with world_size")

        total_cap = sum(int(x) for x in b_max)
        effective_batch = min(target_batch, total_cap)
        if effective_batch <= 0:
            raise ValueError("effective_batch must be positive")

        best_batches: List[int] | None = None
        best_max = float("inf")
        best_var = float("inf")

        def enumerate_batches(prefix: List[int], idx: int, remaining: int) -> None:
            nonlocal best_batches, best_max, best_var
            if idx == self.world_size - 1:
                b_last = remaining
                if b_last < 1 or b_last > b_max[idx]:
                    return
                candidate = prefix + [b_last]
                costs = []
                for rank, b in enumerate(candidate):
                    cost_list = costs_per_rank[rank]
                    if b <= 0 or b > len(cost_list):
                        return
                    cost = cost_list[b - 1]
                    if not math.isfinite(cost):
                        return
                    costs.append(cost)
                max_cost = max(costs)
                if max_cost > best_max:
                    return
                var_cost = float(np.var(costs))
                if max_cost < best_max or (math.isclose(max_cost, best_max) and var_cost < best_var):
                    best_max = max_cost
                    best_var = var_cost
                    best_batches = candidate
                return

            for b in range(1, b_max[idx] + 1):
                if remaining - b < (self.world_size - idx - 1):
                    break
                enumerate_batches(prefix + [b], idx + 1, remaining - b)

        enumerate_batches([], 0, effective_batch)
        if best_batches is None:
            raise RuntimeError("No feasible schedule found")

        return ScheduleResult(
            batches=best_batches,
            max_cost=best_max,
            cost_var=best_var,
            effective_batch=effective_batch,
        )
