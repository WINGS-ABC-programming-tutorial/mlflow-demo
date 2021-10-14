# mlflow-demo
mlflowを使ったパラメータ管理のデモ

## 初期設定
numpy, mlflow, jupyerlaなどをインストール。

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

