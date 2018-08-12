# encoding: utf-8

import os
import json
from bs4 import BeautifulSoup
import re
import codecs
import sys

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

this_file_path = os.path.dirname(__file__)
prj_path = os.path.join(this_file_path, os.pardir)
ask_dir = os.path.join(prj_path, "data/ask_data")
clean_data_dir = os.path.join(prj_path, "data/clean_data")

ask_page_list = os.listdir(ask_dir)


def get_all_page():
    with open("../data/index/index.json", "r") as fp:
        all_index = json.loads(fp.read())
    all_index_url = [_[0].strip(u"：").strip() for _ in all_index]
    return all_index_url


all_department = get_all_page()


def parse_content(a_json):
    a_dict = json.loads(a_json)
    question_title = a_dict.get("question_title")
    question_detail = a_dict.get("question_detail")
    # ask_selected = a_dict.get("ask_selected")

    question_title_soup = BeautifulSoup(question_title, "html5lib")
    question_detail_soup = BeautifulSoup(question_detail, "html5lib")
    # ask_selected_soup = BeautifulSoup(ask_selected, "html5lib")

    title_keys = [_.get_text().strip(u"：") for _ in question_title_soup.find_all("a")]
    title_keys = title_keys[1:] if len(title_keys) > 1 else title_keys
    title_keys = [_ for _ in title_keys if _ in all_department]

    if not title_keys:
        return {}

    ask_tit = question_detail_soup.find_all("p", attrs={"class": "ask_tit"})
    ask_tit_txt = u"。".join([re.sub("\s", "", _.get_text()) for _ in ask_tit])

    mation = question_detail_soup.find_all("p", attrs={"class": "mation"})
    mation_txt = u"。".join([re.sub("\s", "", _.get_text()) for _ in mation])

    ask_hid = question_detail_soup.find_all("div", attrs={"class": "ask_hid"})
    ask_hid_txt = u"。".join([re.sub("\s", "", _.get_text()) for _ in ask_hid])

    return {
        "mation": mation_txt,
        "ask_hid_txt": ask_hid_txt,
        "department": title_keys[-1],
    }


def clean_data():
    for a_f_name in ask_page_list:
        import pdb
        pdb.set_trace()
        print(a_f_name)
        a_f_name_path = os.path.join(ask_dir, a_f_name)
        save_fp = codecs.open(os.path.join(clean_data_dir, a_f_name), "w+", "utf-8")

        with open(a_f_name_path, "r") as fp:
            while True:
                a_line = fp.readline()
                if not a_line:
                    break

                a_question_clean_data = parse_content(a_line.strip())
                if a_question_clean_data:
                    save_fp.write(json.dumps(a_question_clean_data, ensure_ascii=False)+u"\n")

        save_fp.close()

if __name__ == '__main__':
    clean_data()


