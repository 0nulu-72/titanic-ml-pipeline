# titanic-ml-pipeline

[![CI/CD](https://github.com/0nulu-72/titanic-ml-pipeline/actions/workflows/docker.yml/badge.svg)](https://github.com/0nulu-72/titanic-ml-pipeline/actions/workflows/docker.yml)
[![Python Version](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

GitHub Actions で **Docker イメージの自動ビルド & Docker Hub への自動プッシュ** を実装しています。  
公開イメージ : [`0nulu/titanic-ml-pipeline:latest`](https://hub.docker.com/r/0nulu/titanic-ml-pipeline)

Kaggle *Titanic* 生存予測コンペを題材に、前処理 → 学習 → 推論までを自動化した機械学習パイプラインです。

##　🗂️　ディレクトリ構成
```
├── data/
│   ├── train.csv             # 元データ（Kaggleからダウンロード）
│   ├── test.csv              # 元データ（Kaggleからダウンロード）
│   ├── processed_train.pkl   # 前処理後の学習データ
│   └── processed_test.pkl    # 前処理後のテストデータ
├── notebooks/                # Jupyter Notebook ワークフロー
│   └── titanic_workflow.ipynb
├── src/                      # スクリプト版パイプライン
│   ├── preprocess.py         # 前処理: 特徴量エンジニアリング + 入出力
│   ├── train.py              # 学習: パイプライン + モデル訓練 + 保存
│   └── evaluate.py           # 推論: テストデータ予測 + submission.csv出力
├── models/                   # 学習済みモデルを格納
│   └── titanic_model.pkl     # 保存アーティファクト
├── requirements.txt          # Python 依存ライブラリ一覧
├── Dockerfile                # Docker ビルド定義
├── Makefile                  # make all 定義
└── README.md                 # 本ファイル
```

##　⚙️　使い方（２通り）

###　①ローカル実行 ― Python で動かす
以下は Python 3.8+ を想定しています。
```bash
# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt

```


####　📥　データの準備
Kaggle の Titanic コンペページから以下をダウンロードし、data/ フォルダに配置してください。
```bash
mkdir -p data
cp ~/Downloads/train.csv data/
cp ~/Downloads/test.csv  data/
```


####　ワンコマンドでパイプライン実行
```bash
# リポジトリをクローン
git clone https://github.com/0nulu-72/titanic-ml-pipeline.git
cd titanic-ml-pipeline

# 前処理→学習→推論 を一気に
make all
```

####　make&nbsp;all が内部で呼ぶ 3 ステップ
1.前処理 を実行し、中間データを生成
```bash
python src/preprocess.py
```

2.モデル学習 を実行し、学習済みモデルを保存
```bash
python src/train.py
```

3.推論・提出ファイル生成 を実行
```bash
python src/evaluate.py
```
実行後、ルートに submission.csv が生成されます。


###　②　🐳　Docker 実行 ― 依存ゼロ・git clone も不要
```bash
# 1. 最新イメージを取得（ソースコードも含む）
docker pull 0nulu/titanic-ml-pipeline:latest

# 2. data/ をマウントして一連処理を実行
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  0nulu/titanic-ml-pipeline:latest
  # => data/ に processed_*.pkl & submission.csv が出力される
```
ポイント
Pythonもpipなどの環境構築やgit cloneは一切不要。Docker さえ入っていれば上記２行で完了します。

出力物（processed_*.pkl, submission.csv）は data/ に書き出されます。


##　🎯　結果サマリ
| 指標                | 値      |
|---------------------|--------:|
| Public LB スコア     | **0.77990** |
| Validation LogLoss | **0.40490** |
<small>※ Kaggle の非決定性や early stopping の挙動によって若干ブレることがあります。</small>


##　🚀　今後の展望
- 必要に応じて新たな特徴量（例：FarePerPerson, AgeBin, CabinZone など）を追加
- CI/CD を完成させたので、本番環境への自動デプロイ（AWS ECS/EKS, GCP Cloud Run など）を検討
- RandomizedSearchCV を用いた網羅的なハイパーパラメータ探索
- MLflow / Streamlit での実験管理・デモ用 UI の導入

---
このプロジェクトは Kaggle Notebook から派生し、モジュール化したスクリプト版パイプラインとして仕上げたものです。
上記手順をそのまま再現すれば、誰でも同じ結果を得られることを目指しています。

