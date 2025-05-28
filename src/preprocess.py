import pandas as pd

# 1. タイトル抽出とマッピング用の辞書
#    名称(honorific)を共通のカテゴリまたは『Rare』にまとめる
title_map = {
    'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
    'Don': 'Rare', 'Rev': 'Rare', 'Dr': 'Rare', 'Mme': 'Mrs',
    'Ms': 'Miss', 'Major': 'Rare', 'Lady': 'Rare', 'Sir': 'Rare',
    'Mlle': 'Miss', 'Col': 'Rare', 'Capt': 'Rare', 'Countess': 'Rare',
    'Jonkheer': 'Rare', 'Dona': 'Rare'
}

def load_data(train_path: str = 'data/train.csv', test_path: str = 'data/test.csv'):
    """
    2. 生データ(CSV)の読み込み
       - pandas の read_csv で DataFrame として返却
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    3. 特徴量エンジニアリング
       - Name から敬称(Title)を抽出してマッピング
       - 家族構成を表す FamilySize, IsAlone を作成
    """
    # Title を Name 文字列から正規表現で抜き出し
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # title_map によって変換し、未知のものは 'Rare' とする
    df['Title'] = df['Title'].map(title_map).fillna('Rare')

    # 家族数: SibSp + Parch + 自分自身(1)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # 一人かどうかのフラグ
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    4. 欠損値の補完
       - Age, Fare を中央値で埋める
    """
    df['Age']  = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    return df


def preprocess_and_save():
    """
    5. 前処理の実行 & 保存
       1) load_data
       2) feature_engineering + fill_missing を train/test に実行
       3) pickle 形式で保存
    """
    train, test = load_data()

    # 学習データ
    train = feature_engineering(train)
    train = fill_missing(train)

    # テストデータも同一処理
    test  = feature_engineering(test)
    test  = fill_missing(test)

    # downstream 用に pickle 形式で永続化
    train.to_pickle('data/processed_train.pkl')
    test.to_pickle('data/processed_test.pkl')

    print('Processed data saved to data/processed_train.pkl and data/processed_test.pkl')


if __name__ == '__main__':
    preprocess_and_save()
