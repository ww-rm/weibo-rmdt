from dataset import DatasetBuilder
from rmdt import RumorDetector
from wbcmt import WbCmtSpider

if __name__ == "__main__":
    spider = WbCmtSpider("_2AkMX_z1ddcPxrAZTnvwQz2ngbYpH-jykKlSrAn7uJhMyAxhu7n8FqSdutBF-XHa3Zsn7ZaBcRYka1AI6UECM-3hy")
    builder = DatasetBuilder("./data/dict/pretrain_wv.index.json", "./data/dict/pretrain_wv.vec.dat", "cuda")
    detector = RumorDetector("cuda").load("./data/model/rmdt.pt")

    origin, comments = spider.get_comments_by_url("https://weibo.com/1774057271/KfICUisxk")

    raw_input = [origin] + comments
    predict_data = builder.build_input(raw_input)

    result = detector.predict(predict_data)
    print(result["label"])
    print(result["prob"])


