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
        """
        シードを固定して結果が変わっていないことを確認
        """
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

    # シード123と456で試す
    @pytest.mark.parametrize("seed", [123, 456])
    def test_step_linearity_wrt_initial_state(self, seed, init1=-10., init2=+10.):
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
