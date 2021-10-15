# mlflow-demo
mlflowを使ったパラメータ管理のデモ

## 初期設定
numpy, mlflow, jupyerlabなどをインストール。

    pip install -r requirements.txt

VSCodeの場合はDevcontainerで開くと自動でインストールされる。


## 実験方法
(Devcontainerの場合はコンテナ内の)ターミナルを３つ開いて、mlflow, jupyerlab, シミュレーション実行をする。

１つ目はmlflowのサーバーを立ち上げる。

    mlflow server --port 5000

ブラウザで http://127.0.0.1:5000 にアクセスする。Devcontainerを使っていて見れない場合はPort-Forwardingができているか確認する。

２つ目はjupyter labを立ち上げる。

    jupyter lab --port 8888

ブラウザで http://127.0.0.1:8888 にアクセスする。見れない場合は同様にポートの設定を確認。

３つ目のターミナルでシミュレーションのpythonスクリプトを実行する。例えば、

    python simulation3.py 1

引数の数字(`1`)は何個並列計算させるかを指定する。

notebookから実験結果を取得して可視化などを方法は[notebook/demo.ipynb](notebook/demo.ipynb)で紹介した。

## いくつかの実装
ブラウン運動のシミュレーションを例にして4通りの実装を用意した。

 * `simulation1.py`: 一番単純にmlflowを使う
 * `simulation2.py`: 異なるパラメータについて並列で計算
 * `simulation3.py`: クラスを使った実装(`lib3/`)とテスト(`test3/`)
 * `simulation4.py`: mlflowのartifact利用(`lib4/`)とより広範囲なテスト(`test4/`)

## テスト
以下のコマンドを実行する

    pytest
