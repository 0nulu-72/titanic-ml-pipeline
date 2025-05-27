import pandas as pd
import joblib

# 1. Load the trained pipeline
artifacts = joblib.load('models/titanic_model.pkl')
preprocessor = artifacts['preprocessor']
model = artifacts['model']

# 2. Load preprocessed test data
test = pd.read_pickle('data/processed_test.pkl')

# 3. Prepare features (drop unnecessary columns if any)
#    Assuming preprocess.py retained only model features plus PassengerId
X_test = test.drop(columns=['PassengerId','Name','Ticket','Cabin','Embarked'], errors='ignore')

# 4. Predict survival
X_test_proc = preprocessor.transform(X_test)
preds = model.predict(X_test_proc)

# 5. Create submission DataFrame
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': preds.astype(int)
})

# 6. Save to CSV for Kaggle submission
submission.to_csv('submission.csv', index=False)
print(f'Submission saved! ({submission.shape[0]} rows)')
