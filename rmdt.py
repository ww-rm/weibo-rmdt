
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader

import dataset
from model import RumorDetectModel


def eval_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average="macro")
    r = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    return (acc, p, r, f1)


class RumorDetector:
    def __init__(self, device):
        self.model = RumorDetectModel().to(device)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def _train(self, train_loader, criterion, optimizer):
        self.model.train()
        loss_list = []
        pred_list = []
        true_list = []
        for inputs, targets in train_loader:
            # print(targets)
            optimizer.zero_grad()
            outputs, _ = self.model(inputs)  # omit weights
            # print(outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
            true_list.append(targets.cpu().numpy())

        y_pred = np.concatenate(pred_list)
        y_true = np.concatenate(true_list)
        # print(y_pred, y_true)
        loss = np.mean(loss_list)
        acc, p, r, f1 = eval_metrics(y_true, y_pred)

        return (loss, acc, p, r, f1)

    def _eval(self, eval_loader, criterion):
        self.model.eval()
        loss_list = []
        pred_list = []
        true_list = []
        with torch.no_grad():
            for inputs, targets in eval_loader:
                outputs, _ = self.model(inputs)  # omit weights
                loss = criterion(outputs, targets)

                loss_list.append(loss.item())
                pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
                true_list.append(targets.cpu().numpy())

        y_pred = np.concatenate(pred_list)
        y_true = np.concatenate(true_list)

        loss = np.mean(loss_list)
        acc, p, r, f1 = eval_metrics(y_true, y_pred)

        return (loss, acc, p, r, f1)

    def load(self, model_path):
        """从本地加载一个模型"""
        self.model.load_state_dict(torch.load(model_path))
        return self

    def save(self, model_path):
        """保存当前模型参数到本地"""
        torch.save(self.model.state_dict(), model_path)
        return self

    def train(self, train_dataset, valid_dataset, epochs=50, lr=1e-3, b_size=4, log_path=None):
        """训练模型

        参数:
            train_dataset: 通过builder产生
            valid_dataset: 通过builder产生
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr)

        train_loader = DataLoader(train_dataset, b_size, True, collate_fn=dataset.collate_fn)
        valid_loader = DataLoader(valid_dataset, 1, collate_fn=dataset.collate_fn)

        log_content = (
            "*"*50 + "\n" +
            "Epoch: {:02d}\n" +
            "Train Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})\n" +
            "Valid Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})\n" +
            "*"*50 + "\n"
        )
        best_f1 = 0
        for epoch in range(epochs):
            train_loss, train_acc, train_p, train_r, train_f1 = self._train(train_loader, criterion, optimizer)
            valid_loss, valid_acc, valid_p, valid_r, valid_f1 = self._eval(valid_loader, criterion)
            if valid_f1 > best_f1:
                best_f1 = valid_f1
                self.save("./model.tmp")
            log_text = log_content.format(epoch, train_loss, train_acc, train_f1, train_p, train_r, valid_loss, valid_acc, valid_f1, valid_p, valid_r)
            print(log_text)
            if log_path:
                with open(log_path, "a", encoding="utf8") as f:
                    print(log_text, file=f)
        self.load("./model.tmp")  # 读取最优模型
        return self

    def eval(self, eval_dataset, log_path=None):
        """用数据集测试一个模型, 结果输出到log_path

        参数:
            eval_dataset: 通过builder产生
            log_path: 结果输出路径
        """
        eval_loader = DataLoader(eval_dataset, 1, collate_fn=dataset.collate_fn)
        criterion = nn.CrossEntropyLoss()
        eval_loss, eval_acc, eval_p, eval_r, eval_f1 = self._eval(eval_loader, criterion)
        log_content = (
            "*"*50 + "\n" +
            "Eval  Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})\n" +
            "*"*50 + "\n"
        )
        log_text = log_content.format(eval_loss, eval_acc, eval_f1, eval_p, eval_r)
        print(log_text)
        if log_path:
            with open(log_path, "a", encoding="utf8") as f:
                print(log_text, file=f)
        return self

    def update(self, train_dataset, epochs=10, lr=1e-3, log_path=None):
        """用数据集增量更新一个模型

        参数:
            train_dataset: 训练集, 用builder生成
            model_path: 被更新后模型的保存路径
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr)

        train_loader = DataLoader(train_dataset, 1, True, collate_fn=dataset.collate_fn)
        log_content = (
            "*"*50 + "\n" +
            "Epoch: {:02d}\n" +
            "Train Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})\n"
            "*"*50 + "\n"
        )
        for epoch in range(epochs):
            train_loss, train_acc, train_p, train_r, train_f1 = self._train(train_loader, criterion, optimizer)
            log_text = log_content.format(epoch, train_loss, train_acc, train_f1, train_p, train_r)
            print(log_text)
            if log_path:
                with open(log_path, "a", encoding="utf8") as f:
                    print(log_text, file=f)
        return self

    def predict(self, single_input):
        """根据输入返回单个样本的判断结果

        参数:
            single_input: ["text", "text", ...]
            用builder的build_input生成

        返回:
            {
                "label": label, 0: non-rumor, 1: rumor
                "prob": prob,
                "weight": [w, w, w, ...] # 只有评论的权重
            }
        """
        self.model.eval()
        with torch.no_grad():
            outputs, weights = self.model(single_input)
        label = torch.argmax(outputs[0], dim=-1).cpu().numpy()
        prob = outputs[0].cpu().detach().numpy()
        weight = weights[0]

        result = {
            "label": int(label),
            "prob": prob[0] if prob[0] > 0.5 else prob[1],
            "weight": weight
        }
        return result
