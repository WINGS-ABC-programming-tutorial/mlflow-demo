"""
ブラウン運動の実装
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParamBrownianMotion:
    seed: int
    initial_state: float
    sigma: float


class BrownianMotion:

    def __init__(self, param: ParamBrownianMotion) -> None:
        self.initial_state = param.initial_state
        self.sigma = param.sigma
        self.rng = np.random.default_rng(param.seed)

    def step(self, current_state: float) -> float:
        """
        1stepの時間発展を返す

        Parameters
        ----------
        current_state: float

        Returns
        -------
        next_state: float
        """
        return current_state + self.sigma * self.rng.normal()
