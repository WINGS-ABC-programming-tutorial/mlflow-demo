"""
ブラウン運動の実装
"""
from dataclasses import dataclass
from typing import Optional

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
    state: float
    state_trajectory: np.ndarray = np.array([])

    def __init__(
        self,
        param: ParamBrownianMotion,
        save_full_trajectory: bool = False,
        total_step: Optional[int] = None
    ) -> None:
        """
        ブラウン運動

        Parameters
        ----------
        param: ParamBrownianMotion
        save_full_trajectory: bool
            numpy配列として状態の軌跡全てを保持しておく
        total_step: int (optional)
            save_full_trajectoryがTrueのときは必ず指定する
        """
        self.initial_state = param.initial_state
        self.sigma = param.sigma
        self.rng = np.random.default_rng(param.seed)
        self.state = self.initial_state
        self.save_full_trajectory = save_full_trajectory

        if save_full_trajectory:
            if total_step is None:
                raise ValueError("Please set total_step when save_full_trajectory is True")
            # 状態軌跡を保存するnumpy配列の作成と初期化 (長さはtotal_stepに初期状態の分+1)
            self.state_trajectory = np.empty(total_step + 1)
            self.count = 0
            self._save_state()

    def _save_state(self) -> None:
        """
        状態が更新されたときに呼び出す。状態軌跡に情報を書き込む
        """
        if self.save_full_trajectory:
            self.state_trajectory[self.count] = self.state
            self.count += 1
        return

    def step(self) -> float:
        """
        1stepの時間発展を実行して次の状態を返す

        Returns
        -------
        next_state: float
        """
        self.state += self.sigma * self.rng.normal()
        self._save_state()
        return self.state
