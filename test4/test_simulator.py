import tempfile
from pathlib import Path

import numpy as np
import pytest
from lib4.brownian_motion import ParamBrownianMotion
from lib4.simulator import ParamSimulator, Simulator


@pytest.fixture
def mlflow_cache_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir).joinpath("mlruns")
        yield str(cache_dir)


@pytest.fixture
def param_brownian_motion():
    return ParamBrownianMotion(
        seed=123, initial_state=0., sigma=1.
    )


class TestSimulator:
    def test_without_previous_run(self, mlflow_cache_dir, param_brownian_motion):
        total_step = 1000
        record_per = 10
        sim = Simulator(
            exp_name="test",
            param=ParamSimulator(
                total_step=total_step,
                record_per=record_per,
                save_full_traj=True,
                param_bm=param_brownian_motion,
            ),
            cache_dir=mlflow_cache_dir,
            check_previous_runs=True,
        )
        assert not sim.done
        sim.run()

        metric_history = sim.get_metric_history()
        assert len(metric_history) == 1000 // 10 + 1

        state_trajectory = sim.get_state_trajectory()
        assert state_trajectory.shape == (total_step + 1, )
        print(metric_history[:10])
        print(state_trajectory[:21])
        # metric_historyとstate_trajectoryが同じ値をとっているか
        for metric in metric_history:
            assert state_trajectory[metric.step] == metric.value

    def test_with_previous_run(self, mlflow_cache_dir, param_brownian_motion):
        sim1 = Simulator(
            exp_name="test",
            param=ParamSimulator(
                total_step=1000,
                record_per=10,
                save_full_traj=False,
                param_bm=param_brownian_motion,
            ),
            cache_dir=mlflow_cache_dir,
            check_previous_runs=True,
        )
        assert not sim1.done
        sim1.run()
        assert sim1.done

        sim2 = Simulator(
            exp_name="test",
            param=ParamSimulator(
                total_step=1000,
                record_per=10,
                save_full_traj=False,
                param_bm=param_brownian_motion,
            ),
            cache_dir=mlflow_cache_dir,
            check_previous_runs=True,
        )
        assert sim2.done

        # metric_histroyが一致しているか
        metric_history1 = sim1.get_metric_history()
        metric_history2 = sim2.get_metric_history()
        assert len(metric_history1) == len(metric_history2)
        for metric1, metric2 in zip(metric_history1, metric_history2):
            assert metric1.step == metric2.step
            assert metric1.value == metric2.value

        # state_trajectoryが一致しているか
        state_trajectory1 = sim1.get_state_trajectory()
        state_trajectory2 = sim2.get_state_trajectory()
        assert np.allclose(state_trajectory1, state_trajectory2)
