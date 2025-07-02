import pandas as pd
from src.preprocess import load_data, preprocess_and_save

def test_load_data_can_read_train_and_test(tmp_path, monkeypatch):
    # Kaggle データが data/ にある前提なので、
    # テスト用に小さな CSV を tmp_path に置いて monkeypatch する例
    
    # サンプルデータを用意
    sample = pd.DataFrame({'PassengerId':[1], 'Age':[22]})

    # 1) tmp_path 下に data/ フォルダを作成
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # 2) train.csv と test.csv を data/ 配下に書き込み
    sample.to_csv(data_dir / "train.csv", index=False)
    sample.to_csv(data_dir / "test.csv",  index=False)

    # 3) カレントディレクトリを tmp_path に切り替えて読み込み
    monkeypatch.chdir(tmp_path)
    train, test = load_data()

    # 読み込めたことを確認
    assert 'PassengerId' in train.columns
    assert 'Age' in test.columns