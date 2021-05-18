# Weibo Rumor Detection

A small project to automatically crawl micro-blog in Sina Weibo and try to detect whether the specified blog is a rumor.

## Environments

- python >= 3.7
- torch >= 1.6.0
- [requirements.txt](https://github.com/ww-rm/weibo-rmdt/blob/main/requirements.txt)

## Dataset

The dataset used in this project is merged from some small set. It was all pushed into this repo under folder ```data/dataset/raw/```

Use [extractraw.py](https://github.com/ww-rm/weibo-rmdt/blob/main/extractraw.py) to generate ```train```, ```valid``` and ```eval``` datasets.

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

*If you think this project is helpful to you, plz star it and let more people see it. :)*
