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
from pathlib import Path

from joblib import Parallel, cpu_count, delayed

from lib4.brownian_motion import ParamBrownianMotion
from lib4.simulator import ParamSimulator, Simulator

N_seed = 5
x0s = [1.0, -1.0]
sigmas = [0.1, 0.2]


def get_simulator(
    seed: int, x0: float, sigma: float
) -> Simulator:
    """
    Simulatorクラスのインスタンスを作成する

    Parameters
    ----------
    seed: int
    x0: int
    sigma: float

    Returns
    -------
    sim: Simulator
    """
    param = ParamSimulator(
        total_step=500,
        record_per=10,
        save_full_traj=True,
        param_bm=ParamBrownianMotion(
            seed=seed, initial_state=x0, sigma=sigma
        )
    )
    sim = Simulator(
        exp_name="sim4",
        param=param,
        # このファイルがある場所にmlrunsディレクトリをつくる
        #   この指定をするとnotebookからimportしたときにも同じmlrunsを参照できる
        cache_dir=str(Path(__file__).parent.joinpath("mlruns"))
    )
    return sim


def process(*args, **kwargs) -> None:
    """
    シミュレーターを実行するだけの関数
    """
    sim = get_simulator(*args, **kwargs)
    sim.run()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        n_cpus = int(sys.argv[1])
    else:
        n_cpus = cpu_count()

    Parallel(n_jobs=n_cpus)([
        delayed(process)(seed, x0, sigma)
        for seed in range(N_seed)
        for x0 in x0s
        for sigma in sigmas
    ])
