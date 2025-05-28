import pandas as pd

# Title mapping: group honorifics into common or 'Rare'
title_map = {
    'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
    'Don': 'Rare', 'Rev': 'Rare', 'Dr': 'Rare', 'Mme': 'Mrs',
    'Ms': 'Miss', 'Major': 'Rare', 'Lady': 'Rare', 'Sir': 'Rare',
    'Mlle': 'Miss', 'Col': 'Rare', 'Capt': 'Rare', 'Countess': 'Rare',
    'Jonkheer': 'Rare', 'Dona': 'Rare'
}

def load_data(train_path: str = 'data/train.csv', test_path: str = 'data/test.csv'):
    """
    Load raw CSV files into pandas DataFrames.
       Returns (train, test).
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and map passenger titles, then create:
       - FamilySize = SibSp + Parch + 1
       - IsAlone    = 1 if FamilySize == 1 else 0
    """
    # Title extraction from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].map(title_map).fillna('Rare')

    # Family size and alone indicator
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute numeric columns with median:
       - Age, Fare
    """
    df['Age']  = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    return df


def preprocess_and_save():
    """
    Run full preprocessing pipeline:
       - Load data
       - Feature engineering
       - Missing-value imputation
       - Save processed DataFrames as pickle files
    """
    train, test = load_data()

    # Apply steps to train set
    train = feature_engineering(train)
    train = fill_missing(train)

    # Apply same steps to test set
    test  = feature_engineering(test)
    test  = fill_missing(test)

    # Persist for downstream scripts
    train.to_pickle('data/processed_train.pkl')
    test.to_pickle('data/processed_test.pkl')

    print('Processed data saved to data/processed_train.pkl and data/processed_test.pkl')


if __name__ == '__main__':
    preprocess_and_save()
