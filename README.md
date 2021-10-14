# mlflow-demo
mlflowを使ったパラメータ管理のデモ

## 初期設定
numpy, mlflow, jupyerlabなどをインストール。

    pip install -r requirements.txt

VSCodeの場合はDevcontainerで開くと自動でインストールされる。

## 実験方法
(Devcontainerの場合はコンテナ内の)ターミナルを３つ開いて、mlflow, jupyerlab, シミュレーション実行をする。

１つ目はmlflowのサーバーを立ち上げる。

    cd mlflow-demo
    mlflow server --port 5000

ブラウザで http://127.0.0.1:5000 にアクセスする。見れない場合、Devcontainerの場合はPort-Forwardingができているか確認する。

２つ目はjupyter labを立ち上げる。

    jupyter lab --port 8888

ブラウザで http://127.0.0.1:8888 にアクセスする。見れない場合は同様にポートの設定を確認。

３つ目のターミナルでシミュレーションのpythonスクリプトを実行する。

    cd mlflow-demo
    python simulation3.py

## いくつかの実装
ブラウン運動のシミュレーションを例にして3通りの実装を用意しました。

 * `simulation1.py`: 一番単純にmlflowを使う
 * `simulation2.py`: 異なるパラメータについて並列で計算
 * `simulation3.py`: クラスを使った実装+テスト
