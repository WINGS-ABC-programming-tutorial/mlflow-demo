import numpy as np
import pytest
from lib4.brownian_motion import BrownianMotion, ParamBrownianMotion

CORRECT_DATASET = [
    # (seed, initial_state, sigma, state_trajectory)
    (123, 0.0, 10.0, np.array([
        0.,
        -9.891213503,
        -13.56908001,
        -0.68982741,
    ])),
    (456, -2.0, 1.0, np.array([
        -2.,
        -1.103960047,
        -3.227908586,
        -1.587704777,
    ]))
]


class TestBrownianMotion:
    def test_param_negative_sigma_fail(self):
        with pytest.raises(ValueError):
            ParamBrownianMotion(
                seed=123,
                initial_state=0.0,
                sigma=-1.0
            )

    def test_init(self):
        BrownianMotion(ParamBrownianMotion(
            seed=123,
            initial_state=0.0,
            sigma=10.0
        ))

    @pytest.mark.parametrize("seed, initial_state, sigma, state_trajectory", CORRECT_DATASET)
    def test_step(self, seed, initial_state, sigma, state_trajectory):
        bm = BrownianMotion(ParamBrownianMotion(
            seed=seed,
            initial_state=initial_state,
            sigma=sigma
        ))
        assert bm.state == state_trajectory[0]
        for i in range(1, state_trajectory.shape[0]):
            bm.step()
            assert np.isclose(bm.state, state_trajectory[i])

    @pytest.mark.parametrize("seed", [123, 456])
    def test_linearity_of_state_wrt_initial_state(self, seed, init1=-10., init2=+10.):
        """
        シードが同じであれば初期値が違うだけの２つの軌跡は平行移動すれば一致する
        """
        bm1 = BrownianMotion(ParamBrownianMotion(
            seed=seed,
            initial_state=init1,
            sigma=10.0
        ))
        for i in range(100):
            bm1.step()

        bm2 = BrownianMotion(ParamBrownianMotion(
            seed=seed,
            initial_state=init2,
            sigma=10.0
        ))
        for i in range(100):
            bm2.step()

        assert np.isclose(bm1.state - bm2.state, init1 - init2)

    def test_init_save_trajectory_without_total_step_fail(self):
        with pytest.raises(ValueError):
            BrownianMotion(
                param=ParamBrownianMotion(
                    seed=123,
                    initial_state=0.0,
                    sigma=10.0
                ),
                save_full_trajectory=True,
                total_step=None
            )

    @pytest.mark.parametrize("seed, initial_state, sigma, state_trajectory", CORRECT_DATASET)
    def test_step_save_trajectory(self, seed, initial_state, sigma, state_trajectory):
        bm = BrownianMotion(
            param=ParamBrownianMotion(
                seed=seed,
                initial_state=initial_state,
                sigma=sigma
            ),
            save_full_trajectory=True,
            total_step=state_trajectory.shape[0] - 1
        )
        for i in range(state_trajectory.shape[0] - 1):
            bm.step()

        assert np.allclose(bm.state_trajectory, state_trajectory)

    @pytest.mark.parametrize("seed", [123, 456])
    @pytest.mark.parametrize("init1", [-10., +1.])
    @pytest.mark.parametrize("init2", [-10., +10.])
    @pytest.mark.parametrize("sigma", [0.1, 10.])
    def test_linearity_of_trajectory_wrt_initial_state(self, seed, init1, init2, sigma):
        """
        シードが同じであれば初期値が違うだけの２つの軌跡は平行移動すれば一致する
        """
        bm1 = BrownianMotion(ParamBrownianMotion(
            seed=seed,
            initial_state=init1,
            sigma=sigma
        ))
        for i in range(100):
            bm1.step()

        bm2 = BrownianMotion(ParamBrownianMotion(
            seed=seed,
            initial_state=init2,
            sigma=sigma
        ))
        for i in range(100):
            bm2.step()

        assert np.allclose(bm1.state - bm2.state, init1 - init2)
