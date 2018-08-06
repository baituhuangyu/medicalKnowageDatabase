# encoding: utf-8

import requests
from bs4 import BeautifulSoup
import re
import json
import pdb
from functools import reduce
import time
import os


headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36"
}

this_file_path = os.path.dirname(__file__)
prj_path = os.path.join(this_file_path, os.pardir)
page_url_dir = os.path.join(prj_path, "data/page_url")
ask_dir = os.path.join(prj_path, "data/ask_data")

page_url_list = os.listdir(page_url_dir)


def feath_page(session, url):
    try:
        a_index_page = session.get(url, headers=headers, timeout=8).content

        return a_index_page
    except Exception as e:
        print("url error: %s" % url)
        print(e)

    return ""


def get_main_content(html):
    a_soup = BeautifulSoup(html)
    cont_l_tag = a_soup.find("div", attrs={"class": "cont_l"})
    cont_r = cont_l_tag.find("div", attrs={"class": "cont_r"}) if cont_l_tag else ""
    if cont_l_tag:
        cont_r.clear()
    sub_tag = cont_l_tag.find("div", attrs={"id": "sub"}) if cont_l_tag else ""
    ask_cont_tag = cont_l_tag.find("div", attrs={"class": "ask_cont"}) if cont_l_tag else ""

    ask_selected_tag = cont_l_tag.find("div", attrs={"class": "selected"}) if cont_l_tag else ""
    rst_dict = {
        "question_title": str(sub_tag),
        "question_detail": str(ask_cont_tag),
        "ask_selected": str(ask_selected_tag),
    }

    if any(rst_dict.values()):
        return rst_dict
    else:
        return {}


def get_ask_detail_page():
    for a_f_name in page_url_list:
        print(a_f_name)
        a_f_name_path = os.path.join(page_url_dir, a_f_name)
        a_ask_path = os.path.join(ask_dir, a_f_name)
        a_num = 0
        max_num = 10000

        with open(a_f_name_path, "r") as fp:
            while True:
                session = requests.session()

                a_num += 1
                a_line = fp.readline()
                if not a_line or a_num >= max_num:
                    break

                a_question_url = "http://ask.39.net%s" % a_line.strip()
                a_question_page = feath_page(session, a_question_url)
                if a_question_page:
                    a_page_dict = get_main_content(a_question_page)
                    if a_page_dict:
                        a_page_dict["ask_id"] = a_line.strip().replace("/question/", "").replace(".html", "")
                        with open(a_ask_path, "a") as ask_fp:
                            ask_fp.write(json.dumps(a_page_dict, ensure_ascii=False)+"\n")

get_ask_detail_page()

