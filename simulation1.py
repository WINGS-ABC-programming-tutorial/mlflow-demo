#!/usr/bin/env python
"""
Overview:
    パラメータを変えながらブラウン運動のシミュレーションを実行してmlflowに結果を保存する
"""
import itertools

import mlflow
import numpy as np


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
    # 乱数シードを固定する
    rng = np.random.default_rng(seed)
    # mlflowのexperiment nameを設定する
    mlflow.set_experiment("sim1")
    # sim2 Experimentの新しいRunを開始する
    with mlflow.start_run():
        # パラメータを記録
        mlflow.log_params({
            "seed": seed,
            "x0": x0,
            "sigma": sigma,
        })
        x = x0
        for epoch in range(100):
            # x' ~ N(x, sigma^2)
            x = x + sigma * rng.normal()
            # 結果を記録
            mlflow.log_metrics({
                "x": x,
            }, step=epoch)


N_seed = 3
x0s = [1.0, 1.5]
sigmas = [0.1, 0.2]

if __name__ == "__main__":
    for x0, sigma in itertools.product(x0s, sigmas):
        for seed in range(N_seed):
            run_simulation(seed, x0, sigma)
