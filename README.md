Titanic ML Pipeline

このリポジトリは、Kaggle Titanic 生存予測コンペティションを題材にした、機械学習パイプラインを再現・実行可能な形でまとめたものです。

🗂️ ディレクトリ構成
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


⚙️ 環境構築
以下は Python 3.8+ を想定しています。
# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt


📥 データの準備
Kaggle の Titanic - Machine Learning from Disaster から
train.csv と test.csv をダウンロードし、data/ フォルダに配置してください。
cp ~/Downloads/train.csv data/
cp ~/Downloads/test.csv  data/


▶️ 実行手順
前処理 を実行し、中間データを生成
python src/preprocess.py

モデル学習 を実行し、学習済みモデルを保存
python src/train.py

推論・提出ファイル生成 を実行
python src/evaluate.py
これで submission.csv が生成されます。


🎯 結果サマリ
Public LB スコア: 0.78947
Validation LogLoss: 0.40285


🚀 今後の展望
新たな特徴量（FarePerPerson, AgeBin など）の追加
LightGBM / CatBoost とのアンサンブル
RandomizedSearchCV によるハイパーパラメータ探索
CI/CD での自動再現環境構築


このプロジェクトは Kaggle Notebook から派生し、構成をモジュール化したパイプラインとしてまとめられています。

