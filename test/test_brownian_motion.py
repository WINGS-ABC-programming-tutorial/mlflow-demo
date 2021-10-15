import numpy as np
import pytest
from lib3.brownian_motion import BrownianMotion, ParamBrownianMotion


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

    def test_step(self):
        bm = BrownianMotion(ParamBrownianMotion(
            seed=123,
            initial_state=0.0,
            sigma=10.0
        ))
        assert bm.state == 0.0
        bm.step()
        assert np.isclose(bm.state, -9.891213503)
        bm.step()
        assert np.isclose(bm.state, -13.56908001)
