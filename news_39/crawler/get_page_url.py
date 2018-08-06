# encoding: utf-8

import requests
from bs4 import BeautifulSoup
import re
import json
import pdb
from functools import reduce
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36"
}

ok_list = [
    "168-1",
    "229-1",
    "231-1",
    "232-1",
    "27-1",
    "281-1",
    "284-1",
    "304-1",
    "3162-1",
    "3165-1",
    "3166-1",
    "3168-1",
    "319465592-1",
    "4374-1",
    "4376-1",
    "4379-1",
    "4380-1",
    "4381-1",
    "44-1",
]


def get_max_page(html):
    a_soup = BeautifulSoup(html)
    total_page = 0
    # 查找最大page数
    a_left = a_soup.find(attrs={"class": "pgleft"})
    if a_left:
        a_s = a_left.find_all("a")
        if a_s:
            a_s_max_page = re.findall(""".*?-(.*?)\.html""", a_s[-1].get("href"))
            if a_s_max_page and a_s_max_page[0].isdigit():
                total_page = min([int(a_s_max_page[0]), 1000])

    return total_page


def feath_page(session, url):
    try:
        a_index_page = session.get(url, headers=headers, timeout=8).content

        return a_index_page
    except Exception as e:
        print("url error: %s" % url)
        print(e)

    return ""


def get_page_question(html):
    soup = BeautifulSoup(html)
    ul_tag = soup.find("ul", attrs={"class": "list_ask list_ask2"})
    hrefs = []
    if ul_tag:
        a_tags = ul_tag.find_all("a") or []
        hrefs = [_.get("href") for _ in a_tags]
    return hrefs


def get_task_page_url():
    session = requests.session()

    with open("../data/index/index.json", "r") as fp:
        all_index = json.loads(fp.read())

    all_index_url = [_[1][:-2] for _ in all_index]
    all_index_url = list(set(all_index_url))
    all_index_url = filter(lambda _: _ not in ok_list, all_index_url)

    for a_index in all_index_url:
        a_index_pre = a_index.split("-")[0]
        a_url = "http://ask.39.net/news/%s.html" % a_index
        a_index_page_text = "../data/page_url/%s.txt" % a_index
        with open(a_index_page_text, "a") as fp:
            a_index_page = feath_page(session, a_url)

            total_page = get_max_page(a_index_page)
            print("total_page: %s" % total_page)
            a_hrefs = get_page_question(a_index_page)
            print("len(a_hrefs) * total_page: %s", len(a_hrefs) * total_page)

            for line in a_hrefs:
                fp.write(line+"\n")

            if total_page < 2:
                continue

            for i in range(2, total_page):
                a_url_loop = "http://ask.39.net/news/%s-%s.html" % (a_index_pre, i)
                a_page_loop = feath_page(session, a_url_loop)
                a_hrefs_loop = get_page_question(a_page_loop)
                for line_loop in a_hrefs_loop:
                    fp.write(line_loop+"\n")


if __name__ == '__main__':
    get_task_page_url()


