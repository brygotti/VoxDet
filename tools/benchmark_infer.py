#!/usr/bin/env python3
import argparse
import time
from typing import Any

import numpy as np
import torch
from mmcv import Config

from LightningTools.pl_model import pl_model
from LightningTools.dataset_dm import DataModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VoxDet inference latency/memory.")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to benchmark (e.g., 1 2 4 8).",
    )
    parser.add_argument(
        "--measure",
        choices=["model", "e2e", "both"],
        default="model",
        help="model=forward only, e2e=includes data loading + H2D, both=report both.",
    )
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(v, device) for v in batch)
    return batch


def _fetch_batch(data_iter, dataloader):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)
    return batch, data_iter


def _benchmark_once(model, dataloader, device, measure, num_warmup, num_iters):
    data_iter = iter(dataloader)

    for _ in range(num_warmup):
        batch, data_iter = _fetch_batch(data_iter, dataloader)
        batch = _move_to_device(batch, device)
        with torch.no_grad():
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times_ms = []

    if device.type == "cuda":
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(num_iters):
        if measure == "e2e":
            start = time.perf_counter()
            batch, data_iter = _fetch_batch(data_iter, dataloader)
            batch = _move_to_device(batch, device)
            with torch.no_grad():
                model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        else:
            batch, data_iter = _fetch_batch(data_iter, dataloader)
            batch = _move_to_device(batch, device)
            if device.type == "cuda":
                start_evt.record()
                with torch.no_grad():
                    model(batch)
                end_evt.record()
                torch.cuda.synchronize(device)
                elapsed_ms = start_evt.elapsed_time(end_evt)
            else:
                start = time.perf_counter()
                with torch.no_grad():
                    model(batch)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    if device.type == "cuda":
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        peak_alloc_mb = 0.0
        peak_reserved_mb = 0.0

    times_ms = np.asarray(times_ms, dtype=np.float64)
    stats = {
        "mean_ms": float(times_ms.mean()),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p90_ms": float(np.percentile(times_ms, 90)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "peak_alloc_mb": float(peak_alloc_mb),
        "peak_reserved_mb": float(peak_reserved_mb),
    }
    return stats


def main():
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    cfg = Config.fromfile(args.config_path)
    if args.ckpt_path is not None:
        cfg["load_from"] = args.ckpt_path
    cfg.setdefault("save_path", None)
    cfg.setdefault("test_mapping", False)
    cfg.setdefault("pretrain", False)

    model = pl_model(cfg)
    model.eval()
    model.to(device)

    print("Benchmark settings:")
    print(f"  measure={args.measure} warmup={args.num_warmup} iters={args.num_iters} device={device}")

    for bs in args.batch_sizes:
        cfg.test_dataloader_config.batch_size = bs
        dm = DataModule(cfg)
        dm.setup(stage="test")
        dataloader = dm.test_dataloader()

        print("-")
        print(f"batch_size={bs}")

        measures = ["model", "e2e"] if args.measure == "both" else [args.measure]
        for m in measures:
            stats = _benchmark_once(
                model=model,
                dataloader=dataloader,
                device=device,
                measure=m,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
            )

            mean_ms = stats["mean_ms"]
            throughput = bs / (mean_ms / 1000.0)

            print(f"  mode={m}")
            print(
                f"    latency_mean_ms={stats['mean_ms']:.3f}  latency_p50_ms={stats['p50_ms']:.3f} "
                f" latency_p90_ms={stats['p90_ms']:.3f}  latency_p99_ms={stats['p99_ms']:.3f}"
            )
            print(f"    throughput_samples_s={throughput:.2f}")
            print(f"    peak_alloc_mb={stats['peak_alloc_mb']:.1f}  peak_reserved_mb={stats['peak_reserved_mb']:.1f}")


if __name__ == "__main__":
    main()
