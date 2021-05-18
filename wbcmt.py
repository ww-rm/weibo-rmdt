import json
import re
import string
import sys
from pathlib import Path
from time import sleep
from typing import List, Tuple

import bs4
import requests


class base62:
    table_encode = {i: c for i, c in enumerate(string.digits+string.ascii_letters)}
    table_decode = {c: i for i, c in enumerate(string.digits+string.ascii_letters)}

    @staticmethod
    def encode(base10_num: str):
        raise NotImplementedError

    @staticmethod
    def decode(base62_num: str):
        if len(base62_num) % 4 != 0:
            base62_num = "0"*(4 - len(base62_num) % 4) + base62_num

        # print(base62_num)
        result = ""
        for i in range(0, len(base62_num), 4):
            base10_num = str(
                base62.table_decode[base62_num[i+0]]*62**3 +
                base62.table_decode[base62_num[i+1]]*62**2 +
                base62.table_decode[base62_num[i+2]]*62**1 +
                base62.table_decode[base62_num[i+3]]*62**0
            )
            if i > 0 and len(base10_num) < 7:
                base10_num = "0"*(7-len(base10_num)) + base10_num
            result += base10_num

        return result


class WbCmtSpider(requests.Session):
    url_detail = "https://m.weibo.cn/detail/{id_}"
    url_comment_big = "https://weibo.com/aj/v6/comment/big"

    def __init__(self, cookie_SUB: str):
        super().__init__()
        self.headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36 Edg/88.0.705.81",
            "referer": "https://weibo.com/",
        }
        self.cookies.set("SUB", cookie_SUB) # 目前只发现需要这一个 cookie 值

    def requset(self, method, url, *args, **kwargs):
        try:
            sleep(0.1)
            return super().request(method, url, *args, **kwargs)
        except Exception as e:
            print(e)
            return requests.Response()

    def get_status(self, id_) -> str:
        """通过手机端url获取评论原文"""
        response = self.get(self.url_detail.format(id_=id_))
        if response.status_code != 200:
            return ""
        # with open("./cache.html", "w", encoding="utf8", errors="ignore") as f:
        #     f.write(response.text)
        render_json = json.loads(re.findall(r"render_data.*?\[(\{[\S\s]*?\})\]", response.text)[0])
        origin_text = bs4.BeautifulSoup(render_json.get("status").get("text"), "lxml").get_text()
        return origin_text

    def get_comments(self, id_, page_num) -> list:
        """通过id获取某一页评论
        """
        assert page_num >= 1

        response = self.get(self.url_comment_big, params={"id": id_, "page": page_num, "filter": "all"})
        if response.status_code != 200:
            return []
        html_data = response.json().get("data").get("html").replace("\n", "  ").replace("  ", "")
        # with open("./cache_{}.html".format(page_num), "w", encoding="utf8") as f:
        #     f.write(html_data)

        # 获取当前页面的评论内容
        soup = bs4.BeautifulSoup(html_data, "lxml")
        comments = [e.getText().split("：")[1] for e in soup.find_all("div", {"class": "WB_text"})]
        return comments

    def get_status_and_comments(self, id_) -> Tuple[str, List]:
        """返回原文和评论"""
        # 获取原文
        origin = self.get_status(id_)
        # with open("./a.json", "w", encoding="utf8") as f:
        #     json.dump(origin, f, ensure_ascii=False)

        # 获取评论
        page_num = 1
        comments = []
        while True:
            print(page_num)
            _comments = self.get_comments(id_, page_num)
            if not _comments:  # 返回空值停止获取
                break
            comments.extend(_comments)
            page_num += 1

        comments.reverse()  # 反转让时间升序
        # with open("a.json", "w", encoding="utf8") as f:
        #     json.dump(result, f, ensure_ascii=False)
        return (origin, comments)

    def get_id_from_url(self, url) -> str:
        """返回字符串类型数字id"""
        pc_regex = r"weibo.com/[0-9]+?/([0-9a-zA-Z]+)"
        mb_regex = r"m.weibo.cn/status/([0-9a-zA-Z]+)"
        mb_regex2 = r"m.weibo.cn/detail/([0-9]+)"

        id_ = re.findall(pc_regex, url)
        if not id_:
            id_ = re.findall(mb_regex, url)
            if not id_:
                id_ = re.findall(mb_regex2, url)
                if not id_:
                    print("Can't find id from url: {}".format(url), file=sys.stderr)
                    return ""
                else:
                    id_ = id_[0]
            else:
                id_ = base62.decode(id_[0])
        else:
            id_ = base62.decode(id_[0])

        return id_

    def get_comments_by_url(self, comment_url) -> Tuple[str, List]:
        """通过url获取原文和评论, 评论按时间顺序排列"""
        id_ = self.get_id_from_url(comment_url)
        # print(id_)
        return self.get_status_and_comments(id_)


if __name__ == "__main__":
    wbspider = WbCmtSpider("_2A25Np9RbDeRhGeFO6FEQ8CnEzjmIHXVu1UKTrDV8PUNbmtANLRjwkW9NQWDsRD6a218NQq1eERDJT5AakUmjDc3f")
    origin, comments = wbspider.get_comments_by_url("https://weibo.com/1267454277/Kg0WZh0yb")
    print(origin)
    print(comments[-11:-1])
