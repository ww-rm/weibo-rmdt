import json
import os
import random
import shutil
from pathlib import Path

if __name__ == "__main__":
    """用于把原始的json数据集里的评论文本提取出来"""
    dataset_dir = "./data/dataset/"
    raw_dir = Path(dataset_dir, "raw/Weibo")
    raw_label_path = Path(raw_dir, "Weibo.txt")

    all_dir = Path(dataset_dir, "all")
    train_dir = Path(dataset_dir, "train")
    valid_dir = Path(dataset_dir, "valid")
    eval_dir = Path(dataset_dir, "eval")

    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # 处理原始json和标签, 生成all文件夹下的数据集
    all_files = os.listdir(raw_dir)
    for file_name in all_files:
        comment_id, _ = file_name.split(".")
        print(comment_id)
        with open(Path(raw_dir, file_name), "r", encoding="utf8") as f:
            json_ = json.load(f)
            texts = [i["original_text"] for i in json_]
            with open(Path(all_dir, comment_id), "w", encoding="utf8") as f1:
                json.dump(texts, f1, ensure_ascii=False)

    # 生成all下面的标签
    all_labels = {}
    with open(raw_label_path, "r", encoding="utf8") as f:
        for line in f:
            comment_id, comment_label, *_ = line.split("\t")
            _, comment_id = comment_id.split(":")
            _, comment_label = comment_label.split(":")
            all_labels[comment_id] = int(comment_label)
    with open(Path(all_dir, "label.json"), "w", encoding="utf8") as f:
        json.dump(all_labels, f, ensure_ascii=False)

    all_ids = list(all_labels.keys())
    random.seed(1)  # 固定种子
    for _ in range(10000):
        random.shuffle(all_ids)
    total_num = len(all_ids)

    # 生成train
    train_ids = all_ids[0:int(total_num*0.8)]
    for comment_id in train_ids:
        print(comment_id)
        shutil.copyfile(Path(all_dir, comment_id), Path(train_dir, comment_id))
    train_labels = {comment_id: all_labels[comment_id] for comment_id in train_ids}
    with open(Path(train_dir, "label.json"), "w", encoding="utf8") as f:
        json.dump(train_labels, f)

    # 生成valid
    valid_ids = all_ids[int(total_num*0.8):int(total_num*0.9)]
    for comment_id in valid_ids:
        print(comment_id)
        shutil.copyfile(Path(all_dir, comment_id), Path(valid_dir, comment_id))
    valid_labels = {comment_id: all_labels[comment_id] for comment_id in valid_ids}
    with open(Path(valid_dir, "label.json"), "w", encoding="utf8") as f:
        json.dump(valid_labels, f)

    # 生成eval
    eval_ids = all_ids[int(total_num*0.9):total_num]
    for comment_id in eval_ids:
        print(comment_id)
        shutil.copyfile(Path(all_dir, comment_id), Path(eval_dir, comment_id))
    eval_labels = {comment_id: all_labels[comment_id] for comment_id in eval_ids}
    with open(Path(eval_dir, "label.json"), "w", encoding="utf8") as f:
        json.dump(eval_labels, f)
