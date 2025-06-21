
# ---- stage 1: builder ----
# ベースイメージ（Python 3.10 Slim）
FROM python:3.10-slim AS builder

# 作業ディレクトリ
WORKDIR /app

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- stage 2: runtime ----
FROM python:3.10-slim AS runtime
WORKDIR /app
# builder で作った site-packages だけコピー
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .

# デフォルト実行コマンド
# （たとえば前処理→学習→予測まで一気に回すスクリプトがある場合）
CMD ["sh", "-c", "python src/preprocess.py && python src/train.py && python src/evaluate.py"]