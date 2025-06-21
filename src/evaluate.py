import pandas as pd
import joblib
from pathlib import Path

# 1. 保存済み前処理器とモデルを読み込む
artifacts = joblib.load('models/titanic_model.pkl')
preprocessor = artifacts['preprocessor']
model = artifacts['model']

# 2. 前処理されたテストデータを読み込む
#    preprocess.py で生成した pickle ファイルを利用
test = pd.read_pickle('data/processed_test.pkl')

# 3. 予測に使用する特徴量を準備する
#    PassengerId は提出用として残し、
#    Name, Ticket, Cabin, Embarked は削除（errors='ignore' で安全に）
X_test = test.drop(columns=['PassengerId','Name','Ticket','Cabin','Embarked'], errors='ignore')

# 4. 前処理器を使ってテストデータを変換し、モデルで予測を実行
X_test_proc = preprocessor.transform(X_test)
preds = model.predict(X_test_proc)

# 5. Kaggle 形式の提出用 DataFrame を作成
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': preds.astype(int)
})

# 6. CSV ファイルとして保存（インデックスなし）
out_dir = Path("output") 
out_dir.mkdir(exist_ok=True)
submission.to_csv(out_dir / "submission.csv", index=False)
print(f'Submission saved! ({submission.shape[0]} rows)')
