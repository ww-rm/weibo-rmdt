import argparse
import datetime
import random

import numpy as np
import torch

from dataset import DatasetBuilder
from rmdt import RumorDetector


def train(opt):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get builder
    dataset_builder = DatasetBuilder(opt.wv_index_path, opt.wv_path, DEVICE)

    # build dataset
    print("Building dataset...")
    train_dataset = dataset_builder.build_dataset_from_file(opt.train_path)
    valid_dataset = dataset_builder.build_dataset_from_file(opt.valid_path)
    eval_dataset = dataset_builder.build_dataset_from_file(opt.eval_path)

    rumor_detector = RumorDetector(DEVICE)

    print("Start training...")
    rumor_detector.train(train_dataset, valid_dataset, opt.epochs, opt.lr, opt.batch_size, opt.log_path)
    rumor_detector.save(opt.save_path)
    rumor_detector.load(opt.save_path)
    rumor_detector.eval(eval_dataset, opt.log_path)


if __name__ == "__main__":
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data/dataset/train/")
    parser.add_argument("--valid_path", type=str, default="./data/dataset/valid/")
    parser.add_argument("--eval_path", type=str, default="./data/dataset/eval/")
    parser.add_argument("--save_path", type=str, default="./data/model/rmdt.pt")
    parser.add_argument("--log_path", type=str, default="./data/model/report.log")

    parser.add_argument("--wv_index_path", type=str, default="./data/dict/pretrain_wv.index.json")
    parser.add_argument("--wv_path", type=str, default="./data/dict/pretrain_wv.vec.dat")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--seed", type=int, default=1)
    opt = parser.parse_args()
    print(opt)

    # 固定随机种子
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    with open(opt.log_path, "a", encoding="utf8") as f:
        print("*"*10+"Start: "+datetime.datetime.now().strftime("%F %T")+"*"*10, file=f)
        print(opt, file=f)

    # 开始训练
    train(opt)

    with open(opt.log_path, "a", encoding="utf8") as f:
        print("*"*10+"End: "+datetime.datetime.now().strftime("%F %T")+"*"*10+"\n", file=f)
