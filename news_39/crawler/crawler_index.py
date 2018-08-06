# encoding: utf-8

import requests
from bs4 import BeautifulSoup
import re
import json
import pdb
from functools import reduce


def get_task_index():
    session = requests.session()
    url = "http://ask.39.net/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36"
    }

    rst = session.get(url, headers=headers).content

    soup = BeautifulSoup(rst)
    subclassify_1 = soup.find_all(attrs={"class": "page-subclassify-title"})
    subclassify_2 = soup.find_all(attrs={"class": "page-subclassify-item no-border"})
    href_tags = [re.findall("""<a href="/browse/.*?-1-1.*?</a>""", str(_)) for _ in subclassify_1+subclassify_2]
    atags = reduce(lambda _1, _2: _1+_2, href_tags) if href_tags else []

    # atags = re.findall("""<a href="/browse/.*?-1-1.*?</a>""", str(soup))

    index_ = [(re.findall(""">(.*?)<""", _)[0], re.findall("""/browse/(.*?)\.html""", _)[0]) for _ in atags]
    set(index_)

    save_rst = json.dumps(index_, indent=4, ensure_ascii=False)

    with open("../data/index/index.json", "w") as fp:
        fp.write(save_rst)


if __name__ == '__main__':
    get_task_index()


