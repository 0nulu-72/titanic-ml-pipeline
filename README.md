# titanic-ml-pipeline

[![CI](https://github.com/0nulu-72/titanic-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/0nulu-72/titanic-ml-pipeline/actions)
[![Python Version](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

このリポジトリは、Kaggle Titanic 生存予測コンペティションを題材にした、機械学習パイプラインを再現・実行可能な形でまとめたものです。

🗂️ **ディレクトリ構成**
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
├── requirements.txt          # 依存ライブラリ一覧
└── README.md                 # 本ファイル
```

⚙️ **環境構築**
以下は Python 3.8+ を想定しています。
```bash
# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```


📥 データの準備
Kaggle の Titanic コンペページから以下をダウンロードし、data/ フォルダに配置してください。
```bash
cp ~/Downloads/train.csv data/
cp ~/Downloads/test.csv  data/
```


▶️ **実行手順**

**まずはワンコマンドで一気に実行（推奨）**
```bash
make all
```

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


🎯 結果サマリ
- Public LB スコア: 0.77990
- Validation LogLoss: 0.40490
※ Kaggle の非決定性や early stopping の挙動によって若干ブレることがあります。


🚀 今後の展望
- 新たな特徴量（例：FarePerPerson, AgeBin, CabinZone など）の追加
- LightGBM / CatBoost とのアンサンブル・スタッキング
- RandomizedSearchCV を用いた網羅的なハイパーパラメータ探索
- **CI は GitHub Actions で既に構築済み** → 次は**CD（Continuous Deployment）** として、
  Docker コンテナ化＋AWS ECS/EKS または GCP Cloud Run 上への自動デプロイ基盤を整備  
- MLflow / Streamlit などを使った**実験管理・デモ用 UI**の導入

---
このプロジェクトは Kaggle Notebook から派生し、モジュール化したスクリプト版パイプラインとして仕上げたものです。
上記手順をそのまま再現すれば、誰でも同じ結果を得られることを目指しています。

