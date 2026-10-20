"""エピソード単位の処理を複数プロセスで並列に実行する.

CPU 描画（llvmpipe）は 1 プロセスあたりほぼ 1 コアしか使わず、プロセスを増やすと
ほぼ比例して速くなる。各プロセスのスレッド数は 1 に絞り、vCPU の数だけプロセスを立てる。

注意: 子プロセスは spawn で起動し、親の環境変数（MUJOCO_GL など）を引き継ぐ。
親プロセスでは MuJoCo を import する前に :func:`prepare_workers` を呼ぶこと。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any

_SINGLE_THREAD_ENV = {
    "LP_NUM_THREADS": "1",  # llvmpipe（CPU 描画）
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "SVT_LOG": "1",  # AV1 エンコーダのログを警告以上に絞る
}


def prepare_workers(verbose: bool = True):
    """描画バックエンドを選び、子プロセス用の環境変数を設定する. GLInfo を返す."""
    from so101_food.utils.gl import select_gl_backend

    gl = select_gl_backend(verbose=verbose)
    for key, value in _SINGLE_THREAD_ENV.items():
        os.environ.setdefault(key, value)
    return gl


def default_workers() -> int:
    return os.cpu_count() or 1


def run_parallel(
    fn: Callable[[Any], Any],
    jobs: Iterable[Any],
    workers: int,
    label: str = "job",
    max_tasks_per_child: int | None = None,
) -> list[Any]:
    """``fn(job)`` を ``workers`` 個のプロセスで実行し、完了した順に結果を集めて返す.

    ``max_tasks_per_child`` を指定すると、各プロセスはその数のジョブを終えたら作り直される
    （メモリが増え続ける処理向け）.
    """
    jobs = list(jobs)
    results = []
    t0 = time.time()
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            results.append(fn(job))
            print(f"[{label}] {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
        return results
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        max_tasks_per_child=max_tasks_per_child,
    ) as pool:
        futures = [pool.submit(fn, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            print(f"[{label}] {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
    return results
