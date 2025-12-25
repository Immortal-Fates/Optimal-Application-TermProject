from __future__ import annotations

import argparse
import gc
import os
import time
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.data import SampleBuffer, infinite_sharded_iterator, load_sst2
from src.distributed import barrier, setup_distributed
from src.metrics import MetricsWriter, merge_rank_metrics, save_config
from src.scheduler import DynamicBatchScheduler, compute_costs
from src.utils import (
    ensure_dir,
    get_autocast_context,
    get_run_id,
    maybe_create_grad_scaler,
    resolve_precision,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["static", "dynamic"], required=True)
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--global_batch", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=32)
    parser.add_argument("--cost_type", type=str, default="tokens", choices=["tokens", "tokens_sq"])
    parser.add_argument("--precision", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=10000)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--max_local_batch_cap", type=int, default=0)
    parser.add_argument("--probe_max_batch", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    import yaml

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        return args
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def probe_max_batch_size(
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
    max_probe: int,
    precision: str,
) -> int:
    """Estimate max local batch size with synthetic max-length inputs."""
    model.train()
    vocab_size = int(model.config.vocab_size)
    num_labels = int(model.config.num_labels)
    autocast_ctx = get_autocast_context(precision)
    scaler = maybe_create_grad_scaler(precision)

    def try_batch(batch_size: int) -> bool:
        success = True
        try:
            input_ids = torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, max_length),
                device=device,
                dtype=torch.long,
            )
            attention_mask = torch.ones_like(input_ids)
            labels = torch.randint(low=0, high=num_labels, size=(batch_size,), device=device)

            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.update()
            else:
                loss.backward()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                success = False
            else:
                raise
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()
        return success

    last_good = 0
    candidate = 1
    while candidate <= max_probe:
        if try_batch(candidate):
            last_good = candidate
            candidate *= 2
        else:
            break

    if last_good == max_probe:
        return last_good

    low = last_good + 1
    high = min(candidate - 1, max_probe)
    best = last_good
    while low <= high:
        mid = (low + high) // 2
        if try_batch(mid):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return max(best, 1)


def evaluate(
    model: torch.nn.Module,
    data_collator: DataCollatorWithPadding,
    eval_samples: List[Dict],
    device: torch.device,
) -> float:
    """Compute simple accuracy on a small evaluation subset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in eval_samples:
            batch = data_collator([sample])
            batch.pop("length", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            correct += int((preds == batch["labels"]).sum().item())
            total += int(batch["labels"].shape[0])
    model.train()
    return float(correct / max(total, 1))


def main() -> None:
    args = apply_config(parse_args())
    precision = resolve_precision(args.precision)

    local_rank, rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)
    set_seed(args.seed + rank)

    run_id = args.run_id or get_run_id(args.mode)
    run_dir = os.path.join(args.output_dir, run_id)
    metrics_dir = os.path.join(run_dir, "metrics")
    log_dir = os.path.join(run_dir, "logs")

    if rank == 0:
        ensure_dir(metrics_dir)
        ensure_dir(os.path.join(run_dir, "figures"))
        ensure_dir(log_dir)
    barrier()

    logger = setup_logging(rank, log_dir)
    if rank == 0:
        config = vars(args)
        config["precision"] = precision
        save_config(config, os.path.join(run_dir, "config.yaml"))
        with open(os.path.join(args.output_dir, f"last_run_{args.mode}.txt"), "w") as f:
            f.write(run_dir)

    if args.dataset != "sst2":
        raise ValueError("Only sst2 is supported in this project")
    if args.global_batch < world_size:
        raise ValueError("global_batch must be >= world_size")
    if rank == 0:
        ensure_dir(args.dataset_dir)
    barrier()

    train_dataset, tokenizer = load_sst2(
        tokenizer_name=args.model_name,
        split="train",
        max_length=args.max_length,
        seed=args.seed,
        cache_dir=args.dataset_dir,
        max_samples=args.max_train_samples,
    )
    eval_dataset, _ = load_sst2(
        tokenizer_name=args.model_name,
        split="validation",
        max_length=args.max_length,
        seed=args.seed,
        cache_dir=args.dataset_dir,
        max_samples=args.eval_samples,
    )

    train_iter = infinite_sharded_iterator(train_dataset, rank, world_size, args.seed)
    buffer = SampleBuffer(capacity=max(args.buffer_size, 1))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    if rank == 0:
        logger.info("Probing max local batch size per GPU...")
    barrier()
    local_b_max = probe_max_batch_size(
        model=model,
        device=device,
        max_length=args.max_length,
        max_probe=args.probe_max_batch,
        precision=precision,
    )
    if args.max_local_batch_cap > 0:
        local_b_max = min(local_b_max, args.max_local_batch_cap)

    if args.buffer_size < local_b_max and rank == 0:
        logger.warning("buffer_size < local_b_max; consider increasing buffer_size")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.steps
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    scaler = maybe_create_grad_scaler(precision)
    autocast_ctx = get_autocast_context(precision)

    if args.mode == "static":
        if args.global_batch % world_size != 0:
            raise ValueError("global_batch must be divisible by world_size for static mode")
        static_local_batch = args.global_batch // world_size
    else:
        static_local_batch = 0
        batch_scheduler = DynamicBatchScheduler(world_size=world_size, cost_type=args.cost_type)

    fields = [
        "step",
        "rank",
        "local_batch",
        "max_seq_len_in_batch",
        "padded_tokens",
        "step_time_ms",
        "peak_mem_mb",
        "throughput_samples_per_s",
        "throughput_tokens_per_s",
    ]

    metrics_path = os.path.join(metrics_dir, f"rank{rank}.csv")
    with MetricsWriter(metrics_path, fields) as writer:
        buffer.fill(train_iter)
        eval_iter = None
        if args.eval_every > 0 and rank == 0:
            eval_iter = infinite_sharded_iterator(eval_dataset, 0, 1, args.seed)

        for step in range(1, args.steps + 1):
            buffer.fill(train_iter)
            if args.mode == "dynamic":
                local_cap = min(local_b_max, len(buffer))
                length_peek = buffer.peek_lengths(local_cap)

                max_k_t = torch.tensor(len(length_peek), device=device)
                dist.all_reduce(max_k_t, op=dist.ReduceOp.MAX)
                max_k = int(max_k_t.item())

                costs = compute_costs(length_peek, len(length_peek), args.cost_type)
                if len(costs) < max_k:
                    costs.extend([float("inf")] * (max_k - len(costs)))

                costs_t = torch.tensor(costs, device=device)
                gathered_costs = [torch.empty_like(costs_t) for _ in range(world_size)]
                dist.all_gather(gathered_costs, costs_t)

                cap_t = torch.tensor(local_cap, device=device)
                gathered_cap = [torch.empty_like(cap_t) for _ in range(world_size)]
                dist.all_gather(gathered_cap, cap_t)
                b_max_list = [int(t.item()) for t in gathered_cap]

                if rank == 0:
                    costs_per_rank = [t.cpu().tolist() for t in gathered_costs]
                    schedule = batch_scheduler.solve(
                        costs_per_rank=costs_per_rank,
                        b_max=b_max_list,
                        target_batch=args.global_batch,
                    )
                    batch_tensor = torch.tensor(schedule.batches, device=device, dtype=torch.long)
                else:
                    batch_tensor = torch.zeros(world_size, device=device, dtype=torch.long)
                dist.broadcast(batch_tensor, src=0)
                local_batch = int(batch_tensor[rank].item())
            else:
                local_batch = static_local_batch

            batch_samples = buffer.pop(local_batch)
            batch = data_collator(batch_samples)
            batch.pop("length", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            max_seq_len = int(batch["attention_mask"].sum(dim=1).max().item())
            padded_tokens = int(local_batch * max_seq_len)

            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()

            with autocast_ctx:
                outputs = model(**batch)
                loss = outputs.loss

            local_batch_t = torch.tensor(local_batch, device=device, dtype=torch.float32)
            global_batch_t = local_batch_t.clone()
            dist.all_reduce(global_batch_t, op=dist.ReduceOp.SUM)
            global_batch = float(global_batch_t.item())
            scale = world_size * local_batch / max(global_batch, 1.0)
            scaled_loss = loss * scale

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                scaled_loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            torch.cuda.synchronize(device)
            end = time.perf_counter()

            step_time_ms = (end - start) * 1000.0
            peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))

            step_time_t = torch.tensor(step_time_ms, device=device)
            gathered_step_time = [torch.empty_like(step_time_t) for _ in range(world_size)]
            dist.all_gather(gathered_step_time, step_time_t)
            max_step_time_ms = max(float(t.item()) for t in gathered_step_time)

            padded_t = torch.tensor(padded_tokens, device=device, dtype=torch.float32)
            global_padded_t = padded_t.clone()
            dist.all_reduce(global_padded_t, op=dist.ReduceOp.SUM)
            global_padded = float(global_padded_t.item())

            throughput_samples = float(global_batch / max(max_step_time_ms / 1000.0, 1e-6))
            throughput_tokens = float(global_padded / max(max_step_time_ms / 1000.0, 1e-6))

            writer.write(
                {
                    "step": step,
                    "rank": rank,
                    "local_batch": local_batch,
                    "max_seq_len_in_batch": max_seq_len,
                    "padded_tokens": padded_tokens,
                    "step_time_ms": round(step_time_ms, 4),
                    "peak_mem_mb": round(peak_mem_mb, 2),
                    "throughput_samples_per_s": round(throughput_samples, 4),
                    "throughput_tokens_per_s": round(throughput_tokens, 4),
                }
            )

            if rank == 0 and step % args.log_every == 0:
                logger.info(
                    "step=%d local_batch=%d max_step_time=%.2fms throughput=%.2f samples/s",
                    step,
                    local_batch,
                    max_step_time_ms,
                    throughput_samples,
                )

            if args.eval_every > 0 and step % args.eval_every == 0:
                barrier()
                if rank == 0 and eval_iter is not None:
                    eval_samples = [next(eval_iter) for _ in range(args.eval_samples)]
                    acc = evaluate(model.module, data_collator, eval_samples, device)
                    logger.info("eval_step=%d acc=%.4f", step, acc)
                barrier()

    barrier()
    if rank == 0:
        merge_rank_metrics(run_dir, world_size)
        logger.info("Run complete: %s", run_dir)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
