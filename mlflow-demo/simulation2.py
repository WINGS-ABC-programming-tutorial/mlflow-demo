#!/usr/bin/env python
"""
Overview:
    パラメータを変えながらブラウン運動のシミュレーションを実行してmlflowに結果を保存する

Usage:
    simulation.py [<num_cpus>]

Options:
    num_cpus    : 並列数（Default: CPU数)
"""
import sys

import mlflow
import numpy as np
from joblib import Parallel, cpu_count, delayed


def run_simulation(
    seed: int = 0, x0: float = 10, sigma: float = 0.1
) -> None:
    """
    ブラウン運動のシミュレーション

    Parameters
    ----------
    seed: int
        ランダムシード
    x0: float
        初期値
    sigma: float
        ノイズの強さ
    """
    print("seed:", seed, "x0:", x0, "sigma:", sigma)
    rng = np.random.default_rng(seed)
    with mlflow.start_run():
        mlflow.log_params({
            "seed": seed,
            "x0": x0,
            "sigma": sigma,
        })
        x = x0
        for epoch in range(100):
            # x' ~ N(x, sigma^2)
            x = x + sigma * rng.normal()
            mlflow.log_metrics({
                "x": x,
            }, step=epoch)


N_seed = 3
x0s = [1.0, 1.5]
sigmas = [0.1, 0.2]

if __name__ == "__main__":
    if len(sys.argv) == 2:
        n_cpus = int(sys.argv[1])
    else:
        n_cpus = cpu_count()

    Parallel(n_jobs=n_cpus)([
        delayed(run_simulation)(seed, x0, sigma)
        for seed in range(N_seed)
        for x0 in x0s
        for sigma in sigmas
    ])
