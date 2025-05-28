import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# 1. 前処理済み学習データを読み込む
train = pd.read_pickle('data/processed_train.pkl')

# 2. 目的変数(y)と特徴量(X)に分割する
#    Survived を整数型に変換して y に代入
#    不要な列を削除して X を作成
y = train['Survived'].astype(int)
X = train.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])

# 3. 数値型 vs カテゴリ型の列を自動判別するヘルパー関数
def get_numeric_and_categorical(df):
    numeric = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric, categorical

numeric_features, categorical_features = get_numeric_and_categorical(X)

# 4. 前処理パイプラインを構築する
#    数値列: SimpleImputer(strategy='median') で欠損値を中央値で補完
#    カテゴリ列: SimpleImputer(strategy='most_frequent') + OneHotEncoder
numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 5. XGBoost 分類器を定義する
#    ハイパーパラメータは事前にチューニング済み
xgb = XGBClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=2,
    min_child_weight=3,
    reg_lambda=1.0,
    reg_alpha=0.5,
    early_stopping_rounds=80,
    eval_metric='logloss',
    random_state=1,
    n_jobs=1,
    tree_method='exact'
)

# 6. 前処理器とモデルを連結して Pipeline を作成
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb)
])

# 7. 訓練/検証データに分割する (80/20)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

# 8. 前処理器を学習し、NumPy 配列に変換する
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_valid_proc = preprocessor.transform(X_valid)

# 9. 早期打ち切り付きでモデルを学習する
xgb = pipeline.named_steps['classifier']
xgb.fit(
    X_train_proc, y_train,
    eval_set=[(X_train_proc, y_train), (X_valid_proc, y_valid)],
    verbose=False
)

# 10. 検証データにおける最適ラウンドとログロスを表示する
joblib.dump({'preprocessor': preprocessor, 'model': xgb}, 'models/titanic_model.pkl')

# Report best boosting round and validation log-loss
best_it = xgb.best_iteration
best_ll = xgb.evals_result()['validation_1']['logloss'][best_it]
print(f'Best iteration: {best_it}, Validation LogLoss: {best_ll:.5f}')

# 11. 全データで最適ラウンドのみ再学習する
#     n_estimators=best_it に設定し直し、X 全体で再学習
xgb.set_params(n_estimators=best_it, early_stopping_rounds=None)
full_X_proc = preprocessor.transform(X)  # X 全体を NumPy array に変換
xgb.fit(full_X_proc, y, verbose=False)

# 12. 前処理器と再学習済みモデルを保存する
joblib.dump({'preprocessor': preprocessor, 'model': xgb}, 'models/titanic_model.pkl')
print('★ Full-data trained model saved with best_iteration =', best_it)
