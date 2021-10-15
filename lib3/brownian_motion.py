"""
ブラウン運動の実装
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParamBrownianMotion:
    # 乱数シード
    seed: int
    # 初期状態
    initial_state: float
    # ノイズの大きさ(非負実数)
    sigma: float

    def __post_init__(self):
        # 負の値の場合にエラーを出す
        if self.sigma < 0:
            raise ValueError("sigma should be nonnegative")


class BrownianMotion:

    def __init__(self, param: ParamBrownianMotion) -> None:
        self.initial_state = param.initial_state
        self.sigma = param.sigma
        self.rng = np.random.default_rng(param.seed)
        self.state = self.initial_state

    def step(self) -> float:
        """
        1stepの時間発展を実行して次の状態を返す

        Returns
        -------
        next_state: float
        """
        self.state += self.sigma * self.rng.normal()
        return self.state
