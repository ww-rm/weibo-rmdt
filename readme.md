# Weibo Rumor Detection

A small project to automatically crawl micro-blog in Sina Weibo and try to detect whether the specified blog is a rumor.

## Environments

- python >= 3.7
- torch >= 1.6.1
- [requirements.txt](https://github.com/ww-rm/weibo-rmdt/blob/main/requirements.txt)

## Dataset

The dataset used in this project is merged from some small set. It was all uploaded into this repo under folder ```data/dataset/raw/```

Use [extractraw.py](https://github.com/ww-rm/weibo-rmdt/blob/main/extractraw.py) to generate ```train```, ```valid``` and ```eval``` datasets.

## Pretrained Vectors

The raw pretrained vectors is download from repo: [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors) via this link: [Mixed-large 综合 Baidu Netdisk Word + Character + Ngram](https://pan.baidu.com/s/14JP1gD7hcmsWdSpTvA3vKA)

In this project, to avoid huge memory occupation, the raw vectors was processed to a binary data file ```pretrain_wv.vec.dat``` and a index file ```pretrain_wv.index.json```, and use the class ```PretrainedVector``` in dataset.py to load it.
pretrain_wv.indexpretrain_wv.index
You can download ```pretrain_wv.vec.dat``` from the [release page](https://github.com/ww-rm/weibo-rmdt/releases/tag/v1.0.0).

## Train

See [train.py](https://github.com/ww-rm/weibo-rmdt/blob/main/train.py) for details.

After training, it will automatically make evaluation on eval dataset.

## Model

See [model.py](https://github.com/ww-rm/weibo-rmdt/blob/main/model.py) for details.

In this project, it just used fixed parameters to train the model, the parameters of final uploaded [rmdt.pt](https://github.com/ww-rm/weibo-rmdt/blob/main/data/model/rmdt.pt) model is shown in the output below.

```python
RumorDetectModel(
  (origin_bilstm): LSTM(300, 32, batch_first=True, bidirectional=True)
  (comment_lstm): LSTM(300, 64, batch_first=True)
  (comment_dropout): Dropout(p=0.5, inplace=False)
  (attn_U): Linear(in_features=64, out_features=32, bias=False)
  (attn_W): Linear(in_features=64, out_features=32, bias=False)
  (attn_v): Linear(in_features=32, out_features=1, bias=False)
  (linear_dropout): Dropout(p=0.5, inplace=False)
  (linear): Linear(in_features=128, out_features=2, bias=True)
)
```

## Usage

See [main.py](https://github.com/ww-rm/weibo-rmdt/blob/main/main.py) and [rmdt.py](https://github.com/ww-rm/weibo-rmdt/blob/main/rmdt.py) for details.

A simple example is in main.py and [main.ipynb](https://github.com/ww-rm/weibo-rmdt/blob/main/main.ipynb).

## Known Problem

Due to model limitations, the input data must have both original blog text and at least one comment text, otherwise may throw exceptions.

---

*If you think this project is helpful to you, plz star it and let more people see it. :)*
