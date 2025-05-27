# src/preprocess.py
import pandas as pd

# Title mapping based on titanic_workflow.ipynb logic
title_map = {
    'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
    'Don': 'Rare', 'Rev': 'Rare', 'Dr': 'Rare', 'Mme': 'Mrs',
    'Ms': 'Miss', 'Major': 'Rare', 'Lady': 'Rare', 'Sir': 'Rare',
    'Mlle': 'Miss', 'Col': 'Rare', 'Capt': 'Rare', 'Countess': 'Rare',
    'Jonkheer': 'Rare', 'Dona': 'Rare'
}

def load_data(train_path: str = 'data/train.csv', test_path: str = 'data/test.csv'):
    """
    Load raw Titanic CSV files into DataFrames.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform exactly the same feature steps as in titanic_workflow.ipynb:
      - Extract Title from Name and map rare titles
      - Compute FamilySize and IsAlone
    """
    # Extract Title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].map(title_map).fillna('Rare')

    # Family and Alone features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing Age and Fare with median, matching notebook.
    """
    df['Age']  = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    return df


def preprocess_and_save():
    """
    Run full pipeline and save processed data to pickle, mirroring notebook steps.
    """
    train, test = load_data()

    # Apply features and imputation
    train = feature_engineering(train)
    train = fill_missing(train)

    test  = feature_engineering(test)
    test  = fill_missing(test)

    # Save to pickle for downstream scripts
    train.to_pickle('data/processed_train.pkl')
    test.to_pickle('data/processed_test.pkl')

    print('Processed data saved to data/processed_train.pkl and data/processed_test.pkl')


if __name__ == '__main__':
    preprocess_and_save()
