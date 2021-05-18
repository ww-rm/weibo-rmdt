import json
import struct
from pathlib import Path

import jieba
import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainedVector:
    def __init__(self, wv_index_path, wv_path):
        self.index_table = None
        self.wv_dict = None
        with open(wv_index_path, "r", encoding="utf8") as f:
            self.index_table = json.load(f)
        self.wv_dict = open(wv_path, "rb")

    def __del__(self):
        self.wv_dict.close()

    def __getitem__(self, item):
        index = self.index_table.get(item)
        if index is None:
            return None
        else:
            self.wv_dict.seek(4*300*index, 0)
            vec = struct.unpack("f"*300, self.wv_dict.read(1200))
            return np.array(vec)

    def get(self, item, default=None):
        value = self[item]
        if value is None:
            return default
        else:
            return value


class WeiboDataset(Dataset):
    def __init__(self, inputs, targets, device):
        """
        inputs: [
            [length*300的原文, n*300的压缩后的评论],
            ...
        ]
        targets: [
            [0], [1], ...
        ]
        """
        self.inputs = inputs
        self.targets = targets
        self.device = device

    def __getitem__(self, index):
        # print(self.inputs[index])
        return (
            [torch.FloatTensor(self.inputs[index][0]).to(self.device),
             torch.FloatTensor(self.inputs[index][1]).to(self.device)],
            torch.LongTensor(self.targets[index]).to(self.device)
        )

    def __len__(self):
        return len(self.inputs)


def collate_fn(data):
    inputs, targets = map(list, zip(*data))

    inputs = list(map(list, zip(*inputs)))
    # [
    #   [l1*300, l2*300, ...],
    #   [n1*300, n2*300, ...]
    # ]
    targets = torch.cat(targets, dim=0)

    return (inputs, targets)


class DatasetBuilder:
    def __init__(self, wv_index_path, wv_path, device):
        self.wv = PretrainedVector(wv_index_path, wv_path)
        self.device = device

    def _build_one(self, texts):
        """构造一篇微博

        参数:
            texts: ["text", "text", ...]

        返回:
            [length*300的原文, n*300的压缩后的评论]
        """
        input_ = []
        texts = [jieba.lcut(i) for i in texts]  # 分词
        input_.append([self.wv.get(word, np.zeros(300)) for word in texts[0]])  # 转换原文, 二维, length*300

        # 转换评论
        input_.append([])
        for text in texts[1:]:
            input_[1].append(np.mean([self.wv.get(word, np.zeros(300)) for word in text], axis=0))

        return input_

    def build_dataset(self, texts, labels):
        """
        参数:
            texts: [["t", "t", ...], ["t", "t", ...], ...]
            labels: [0, 1, ...]
        返回:
            需要用Dataloader
        """
        inputs = []
        targets = []

        for input_, target in zip(texts, labels):
            inputs.append(self._build_one(input_))  # 分开原文和评论
            targets.append([int(target)])  # 要加一个括号
        dataset = WeiboDataset(inputs, targets, self.device)
        return dataset

    def build_dataset_from_file(self, dataset_dir):
        """从文件中创建数据集

        返回:
            需要用Dataloader
        """
        with open(Path(dataset_dir, "label.json"), "r") as f:
            label_json = json.load(f)

        texts = []
        labels = []
        for id_, label in label_json.items():
            with open(Path(dataset_dir, id_), "r", encoding="utf8") as f:
                texts.append(json.load(f))
            labels.append(label)

        return self.build_dataset(texts, labels)

    def build_input(self, texts):
        """获得适合模型的单个输入

        参数:
            texts: ["text", "text", ...]
        返回:
            直接进模型
        """
        texts = self._build_one(texts)
        return [[torch.FloatTensor(texts[0]).to(self.device)],
                [torch.FloatTensor(texts[1]).to(self.device)]]
