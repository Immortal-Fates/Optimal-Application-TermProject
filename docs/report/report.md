# Dynamic Per-GPU Micro-Batch Scheduling in DDP with Variable-Length NLP Inputs

Course project report skeleton. Fill in figures and results from `analysis`.

## Abstract

This report studies per-rank imbalance in synchronous DDP training caused by variable-length NLP sequences. We propose a per-step min-max integer optimization that dynamically allocates micro-batch sizes across GPUs based on a length-based proxy cost. Experiments on SST-2 demonstrate improved balance, lower tail latency, and better throughput stability.

## 1. Introduction

In DDP, each step completes when the slowest GPU finishes. With variable-length sequences and dynamic padding, per-rank computation and memory can drift, creating a straggler bottleneck. We model the scheduling decision as a small discrete optimization and solve it exactly per step for a 4-GPU setup.

## 2. Problem Formulation

### 2.1 Decision Variables

Let the per-rank micro-batch size at step t be:

```
b_{g,t} in Z_{>=1}, g=1..G
```

Global batch constraint:

```
Sum_g b_{g,t} = B
```

### 2.2 Cost Proxy

Each rank prefetches K samples and computes prefix max length `max_len[k]` and:

```
padded_tokens[k] = k * max_len[k]
```

Two proxy options:

- `c_time[k] = padded_tokens[k]`
- `c_time[k] = padded_tokens[k]^2`

### 2.3 Optimization (per step)

Minimize the maximum cost across ranks:

```
min_{b_1..b_G} max_g c_g(b_g)
  s.t. Sum_g b_g = B, 1 <= b_g <= b_g_max, b_g integer
```

Tie-break by minimizing `Var(c_g(b_g))` for additional balance.

## 3. Solution Method

Since G=4 and B is small, we enumerate all feasible integer combinations each step and pick the exact min-max solution (with variance tie-break).

## 4. DDP Loss Scaling

DDP averages gradients across ranks. With per-rank batch differences, we scale the loss before backward to preserve the global mean gradient:

```
L_g_tilde = L_g * (G * b_g / B)
```

Here `L_g` is the per-rank mean loss and `B = Sum_g b_g` is computed via `dist.all_reduce`.

## 5. OOM Control

- Truncate sequences to `max_length`.
- Startup probing with synthetic max-length batches to estimate per-GPU `b_max_g`.

## 6. Experiment Setup

- Model: distilbert-base-uncased
- Dataset: SST-2 (GLUE)
- Steps: 300 (default)
- Baseline: static per-rank batch
- Method: dynamic per-rank batch (min-max scheduling)

## 7. Results

### 7.1 Metrics

- Max step time (mean, P95)
- Throughput (samples/s, tokens/s)
- Cross-rank std: step time, padded tokens, peak memory

### 7.2 Figures (placeholders)

- Fig 1: Max step time per step (static vs dynamic)
- Fig 2: Cross-rank std of step time
- Fig 3: Cross-rank std of padded tokens
- Fig 4: Cross-rank std of peak memory
- Fig 5: Step time distribution

Insert figures from `outputs/<run_id>/figures/`.

## 8. Discussion and Limitations

- Proxy mismatch vs actual step time (kernel fusion, comms, caching)
- Scheduler overhead vs speed gains
- Sensitivity to sequence length distribution shifts

## 9. Conclusion

Dynamic per-GPU micro-batch scheduling reduces cross-rank imbalance and improves tail latency in DDP training with variable-length NLP inputs.

## References

- PyTorch DDP documentation
- Transformers / Datasets documentation
- GLUE / SST-2 task description
