"""
ブラウン運動シミュレータ
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import mlflow
from flatten_dict import flatten, unflatten

from .brownian_motion import BrownianMotion, ParamBrownianMotion


@dataclass(frozen=True)
class ParamSimulator:
    total_step: int = 1000
    record_per: int = 10
    param_bm: ParamBrownianMotion = ParamBrownianMotion(
        seed=0, initial_state=0., sigma=1.)


class Simulator:
    def __init__(
        self,
        exp_name: str,
        param: ParamSimulator,
        cache_dir: str = "./mlruns",
        run_name: Optional[str] = None,
        run_tags: Optional[Dict[str, Any]] = None,
        check_previous_runs: bool = True,
    ) -> None:
        self.cache_dir = cache_dir
        self.total_step = param.total_step
        self.record_per = param.record_per
        self.bm = BrownianMotion(param.param_bm)

        # get flattened dict of params
        self.params_mlflow = flatten(asdict(param), reducer='dot')

        # mlflow setup
        mlflow.set_tracking_uri(self.cache_dir)
        self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=self.cache_dir)
        exp = self.mlflow_client.get_experiment_by_name(exp_name)
        if exp is not None:
            self.exp_id = exp.experiment_id
        else:
            self.exp_id = self.mlflow_client.create_experiment(exp_name)
        self.run_tags = run_tags
        self.run_name = run_name

        self.done = False
        self.result: Dict[str, Any] = {}
        self.run_id: Optional[str] = None
        if check_previous_runs:
            # check if there is already a finished result with the same parameters
            query = " and ".join([f"param.{k} = '{v}'" for k, v in self.params_mlflow.items()])
            query += " and attributes.status = 'FINISHED'"
            df_result = mlflow.search_runs(
                experiment_ids=[self.exp_id],
                filter_string=query,
                max_results=1,
            )
            if len(df_result) > 0:
                self.done = True
                # convert the pandas DataFrame to an unflattened dict
                self.result = unflatten(df_result.iloc[0].to_dict(), splitter="dot")
                self.run_id = self.result["run_id"]

    def run(self) -> None:
        """
        シミュレーションを１試行実行する
        """
        if self.done:
            print("Simulation already finished!")
            return

        with mlflow.start_run(
            experiment_id=self.exp_id,
            run_name=self.run_name,
            tags=self.run_tags
        ) as run:
            self.run_id = run.info.run_id
            print(f"Starting Run {self.run_name} (ID={self.run_id})")
            mlflow.log_params(self.params_mlflow)

            # initialize
            state = self.bm.initial_state
            mlflow.log_metrics({
                "state": state,
            }, step=0)
            # start simulation
            for step in range(self.total_step):
                state = self.bm.step(state)

                if step % self.record_per == self.record_per - 1:
                    mlflow.log_metrics({
                        "state": state,
                    }, step=step)
        self.done = True
        return

    def get_state_history(self) -> List[mlflow.entities.Metric]:
        if self.run_id is None:
            raise RuntimeError("Please run simulation first or set the params of finished result")

        return self.mlflow_client.get_metric_history(self.run_id, "state")
