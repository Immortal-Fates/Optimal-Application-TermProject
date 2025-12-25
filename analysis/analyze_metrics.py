from __future__ import annotations

import argparse
import os
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def load_merged(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "merged_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing merged_metrics.csv in {run_dir}")
    return pd.read_csv(path)


def aggregate_step_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("step")
    agg = pd.DataFrame(
        {
            "max_step_time_ms": grouped["step_time_ms"].max(),
            "std_step_time_ms": grouped["step_time_ms"].std(ddof=0),
            "std_padded_tokens": grouped["padded_tokens"].std(ddof=0),
            "std_peak_mem_mb": grouped["peak_mem_mb"].std(ddof=0),
            "throughput_samples_per_s": grouped["throughput_samples_per_s"].first(),
            "throughput_tokens_per_s": grouped["throughput_tokens_per_s"].first(),
        }
    )
    return agg.fillna(0.0)


def summarize(agg: pd.DataFrame) -> Dict[str, float]:
    summary = {
        "mean_max_step_time_ms": float(agg["max_step_time_ms"].mean()),
        "p95_max_step_time_ms": float(agg["max_step_time_ms"].quantile(0.95)),
        "mean_throughput_samples_per_s": float(agg["throughput_samples_per_s"].mean()),
        "p95_throughput_samples_per_s": float(agg["throughput_samples_per_s"].quantile(0.95)),
        "mean_throughput_tokens_per_s": float(agg["throughput_tokens_per_s"].mean()),
        "p95_throughput_tokens_per_s": float(agg["throughput_tokens_per_s"].quantile(0.95)),
        "mean_std_step_time_ms": float(agg["std_step_time_ms"].mean()),
        "mean_std_padded_tokens": float(agg["std_padded_tokens"].mean()),
        "mean_std_peak_mem_mb": float(agg["std_peak_mem_mb"].mean()),
    }
    return summary


def plot_compare(
    agg_a: pd.DataFrame,
    agg_b: pd.DataFrame | None,
    label_a: str,
    label_b: str | None,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def line_plot(metric: str, title: str, filename: str, ylabel: str) -> None:
        plt.figure(figsize=(8, 4))
        plt.plot(agg_a.index.to_numpy(), agg_a[metric].to_numpy(), label=label_a)
        if agg_b is not None and label_b:
            plt.plot(agg_b.index.to_numpy(), agg_b[metric].to_numpy(), label=label_b)
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()

    line_plot(
        "max_step_time_ms",
        "Max Step Time per Step",
        "max_step_time.png",
        "ms",
    )
    line_plot(
        "std_step_time_ms",
        "Std of Step Time across Ranks",
        "std_step_time.png",
        "ms",
    )
    line_plot(
        "std_padded_tokens",
        "Std of Padded Tokens across Ranks",
        "std_padded_tokens.png",
        "tokens",
    )
    line_plot(
        "std_peak_mem_mb",
        "Std of Peak Memory across Ranks",
        "std_peak_mem_mb.png",
        "MB",
    )

    plt.figure(figsize=(6, 4))
    plt.hist(agg_a["max_step_time_ms"].to_numpy(), bins=30, alpha=0.6, label=label_a)
    if agg_b is not None and label_b:
        plt.hist(agg_b["max_step_time_ms"].to_numpy(), bins=30, alpha=0.6, label=label_b)
    plt.xlabel("max step time (ms)")
    plt.ylabel("count")
    plt.title("Step Time Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step_time_distribution.png"), dpi=150)
    plt.close()


def build_summary_table(
    summary_a: Dict[str, float],
    summary_b: Dict[str, float] | None,
    label_a: str,
    label_b: str | None,
) -> pd.DataFrame:
    rows = []
    row_a = {"run": label_a, **summary_a}
    rows.append(row_a)
    if summary_b is not None and label_b:
        row_b = {"run": label_b, **summary_b}
        rows.append(row_b)

    df = pd.DataFrame(rows)
    if summary_b is not None and label_b:
        improvements = {"run": f"{label_b}_vs_{label_a}"}
        for key in summary_a.keys():
            base = summary_a[key]
            new = summary_b[key]
            if base == 0:
                improvements[key] = 0.0
            else:
                improvements[key] = (new - base) / base * 100.0
        df = pd.concat([df, pd.DataFrame([improvements])], ignore_index=True)
    return df


def write_markdown_table(df: pd.DataFrame, path: str) -> None:
    headers = [str(h) for h in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = [str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--compare_dir", type=str, default=None)
    parser.add_argument("--run_label", type=str, default=None)
    parser.add_argument("--compare_label", type=str, default=None)
    args = parser.parse_args()

    run_label = args.run_label or os.path.basename(args.run_dir.rstrip("/"))
    compare_label = None
    if args.compare_dir:
        compare_label = args.compare_label or os.path.basename(args.compare_dir.rstrip("/"))

    df_a = load_merged(args.run_dir)
    agg_a = aggregate_step_metrics(df_a)
    summary_a = summarize(agg_a)

    agg_b = None
    summary_b = None
    if args.compare_dir:
        df_b = load_merged(args.compare_dir)
        agg_b = aggregate_step_metrics(df_b)
        summary_b = summarize(agg_b)

    out_dir = os.path.join(args.run_dir, "figures")
    plot_compare(agg_a, agg_b, run_label, compare_label, out_dir)

    summary_df = build_summary_table(summary_a, summary_b, run_label, compare_label)
    summary_csv = os.path.join(args.run_dir, "summary.csv")
    summary_md = os.path.join(args.run_dir, "summary.md")
    summary_df.to_csv(summary_csv, index=False)
    write_markdown_table(summary_df, summary_md)


if __name__ == "__main__":
    main()
