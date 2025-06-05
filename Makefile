# ---------- Titanic ML Pipeline : Make targets ----------
.PHONY: all preprocess train evaluate clean

PYTHON ?= python

# ワンコマンド実行
all: preprocess train evaluate

preprocess:
	$(PYTHON) src/preprocess.py

train: preprocess
	$(PYTHON) src/train.py

evaluate: train
	$(PYTHON) src/evaluate.py

clean:
	rm -f submission.csv
