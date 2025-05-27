import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# 1. Load preprocessed data
train = pd.read_pickle('data/processed_train.pkl')

# 2. Separate features and target
y = train['Survived'].astype(int)
X = train.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])

# If Title or other columns present, include them automatically
def get_numeric_and_categorical(df):
    numeric = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric, categorical

numeric_features, categorical_features = get_numeric_and_categorical(X)

# 3. Build preprocessing pipeline
numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4. Define model with tuned parameters
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
    n_jobs=-1
)

# 5. Create full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb)
])

# 6. Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

# 7. 前処理だけ先に適用（NumPy array に変換）
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_valid_proc = preprocessor.transform(X_valid)

# 8. XGBoost モデルを直接 fit（early stopping 用の eval_set に NumPy 配列を渡す）
xgb = pipeline.named_steps['classifier']
xgb.fit(
    X_train_proc, y_train,
    eval_set=[(X_train_proc, y_train), (X_valid_proc, y_valid)],
    verbose=False
)

# 9. パイプライン全体を保存する場合は、
#    前処理器と学習済みモデルをまとめて pickle するか、
#    再度 Pipeline 経由で joblib.dump してください。
joblib.dump({'preprocessor': preprocessor, 'model': xgb}, 'models/titanic_model.pkl')

# 10. 最適ラウンドと logloss を確認
best_it = xgb.best_iteration
best_ll = xgb.evals_result()['validation_1']['logloss'][best_it]
print(f'Best iteration: {best_it}, Validation LogLoss: {best_ll:.5f}')
