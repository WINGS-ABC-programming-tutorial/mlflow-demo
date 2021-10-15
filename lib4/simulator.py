"""
ブラウン運動シミュレータ
"""
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
from flatten_dict import flatten, unflatten

from .brownian_motion import BrownianMotion, ParamBrownianMotion


@dataclass(frozen=True)
class ParamSimulator:
    # 何ステップの時間発展をするか
    total_step: int = 1000
    # 何ステップおきにmlflowに記録するか
    record_per: int = 10
    # 軌跡全部をmlflow artifact全体として保存するか
    save_full_traj: bool = True
    # ブラウン運動のパラメータ
    param_bm: ParamBrownianMotion = ParamBrownianMotion(
        seed=0, initial_state=0., sigma=1.)


class Simulator:
    def __init__(
        self,
        exp_name: str,  # mlflowの実験の名前
        param: ParamSimulator,
        cache_dir: str = "./mlruns",  # mlflowのデータ保存先
        run_name: Optional[str] = None,  # mlflowのRunにつける名前
        run_tags: Optional[Dict[str, Any]] = None,  # mlflowのRunにつけるタグ
        check_previous_runs: bool = True,  # 同じパラメータでの実験結果がないか検索する
    ) -> None:
        self.cache_dir = cache_dir
        self.total_step = param.total_step
        self.record_per = param.record_per
        self.save_full_trajectory = param.save_full_traj
        self.bm = BrownianMotion(param.param_bm, param.save_full_traj, self.total_step)

        # パラメータをflattenした辞書として取得する
        #   flattenすることでmlflowが受け取ってくれる
        #   パラメータに階層構造があってもドットでつなげてくれる
        self.params_mlflow = flatten(asdict(param), reducer='dot')

        # mlflowをセットアップする
        mlflow.set_tracking_uri(self.cache_dir)
        self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=self.cache_dir)
        exp = self.mlflow_client.get_experiment_by_name(exp_name)
        if exp is not None:
            self.exp_id = exp.experiment_id
        else:
            self.exp_id = self.mlflow_client.create_experiment(exp_name)
        self.run_tags = run_tags
        self.run_name = run_name

        # シミュレーション実行後に結果を取得できるようにする準備
        #   check_previous_runs=Trueなら過去の結果をmlflowから取り出す
        self.done = False
        self.result: Dict[str, Any] = {}
        self.run_id: Optional[str] = None
        if check_previous_runs:
            # 同じパラメータでFINISHEDステータスになっている結果があるか検索する
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
        # すでに実行済みの場合は実行しない
        if self.done:
            print("Simulation already finished!")
            return

        # mlflowのRunを開始する
        with mlflow.start_run(
            experiment_id=self.exp_id,
            run_name=self.run_name,
            tags=self.run_tags
        ) as run:
            self.run_id = run.info.run_id
            print(f"Starting Run {self.run_name} (ID={self.run_id})")
            print(self.params_mlflow)
            mlflow.log_params(self.params_mlflow)

            # 初期化
            state = self.bm.state
            mlflow.log_metrics({
                "state": state,
            }, step=0)
            # シミュレーション開始
            for step in range(self.total_step):
                state = self.bm.step()

                if step % self.record_per == self.record_per - 1:
                    mlflow.log_metrics({
                        "state": state,
                    }, step=step)

            # 状態軌跡をmlflowにartifactとして保存
            self.state_trajectory = self.bm.state_trajectory
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir).joinpath("state_trajectory.bin")
                with tmp_path.open("wb") as f:
                    np.save(f, self.state_trajectory)
                mlflow.log_artifact(str(tmp_path))

        self.done = True
        return

    def get_metric_history(self) -> List[mlflow.entities.Metric]:
        """
        シミュレーションを実行したあとで状態の軌跡を取得する
        run_idが必要なので、check_previous_run=Trueでコンストラクタを呼び出すか
        run()を実行したあとでないとRuntimeErrorを発生させる
        これはrecord_perステップおきに保存されたものを取得する。軌跡全体を取得するにはget_state_trajectory()

        Returns
        -------
        state_history: List[mlflow.entries.Metric]
        """
        if self.run_id is None:
            raise RuntimeError("Please run simulation first or set the params of finished result")

        return self.mlflow_client.get_metric_history(self.run_id, "state")

    def get_state_trajectory(self) -> np.ndarray:
        """
        シミュレーションを実行したあとで状態の軌跡全体を取得する
        record_perステップおきにmetricとして保存されたものを取得するにはget_metric_histroy()

        Returns
        -------
        state_trajectory: np.ndarray (total_step, ) float
        """
        # run()でシミュレーションを実行した後なら実行結果のデータがすでにある
        if self.state_trajectory is not None:
            return self.state_trajectory
        else:
            # Use download_artifacts if artifacts are not saved in the local filesystem
            parent_path = Path(self.cache_dir).parent
            artifact_path = parent_path.joinpath(
                self.result["artifact_uri"],
                "state_trajectory.bin"
            )
            if artifact_path.exists():
                data = np.load(str(artifact_path))
                self.state_trajectory = data
                return data
            else:
                raise FileNotFoundError(f"{str(artifact_path)} does not exists!")
