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


def parse_sex(text):
    if "女" in text:
        return "女"
    elif "男" in text:
        return "男"
    else:
        return ""


common_used_numerals_tmp = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                            '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
common_used_numerals = {}
for key in common_used_numerals_tmp:
    common_used_numerals[key] = common_used_numerals_tmp[key]


def chinese2digits(uchars_chinese):
    total = 0
    r = 1  # 表示单位：个十百千...
    for i in range(len(uchars_chinese) - 1, -1, -1):
        val = common_used_numerals.get(uchars_chinese[i])
        if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
            if val > r:
                r = val
                total = total + val
            else:
                r = r * val
                # total =total + r * x
        elif val >= 10:
            if val > r:
                r = val
            else:
                r = r * val
        else:
            total = total + r * val
    return total


num_str_start_symbol = ['一', '二', '两', '三', '四', '五', '六', '七', '八', '九',
                        '十']
more_num_str_symbol = ['零', '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']


def changeChineseNumToArab(oriStr):
    lenStr = len(oriStr)
    aProStr = ''
    if lenStr == 0:
        return aProStr

    hasNumStart = False
    numberStr = ''
    for idx in range(lenStr):
        if oriStr[idx] in num_str_start_symbol:
            if not hasNumStart:
                hasNumStart = True

            numberStr += oriStr[idx]
        else:
            if hasNumStart:
                if oriStr[idx] in more_num_str_symbol:
                    numberStr += oriStr[idx]
                    continue
                else:
                    numResult = str(chinese2digits(numberStr))
                    numberStr = ''
                    hasNumStart = False
                    aProStr += numResult

            aProStr += oriStr[idx]
            pass

    if len(numberStr) > 0:
        resultNum = chinese2digits(numberStr)
        aProStr += str(resultNum)

    return aProStr


def parse_age(text):
    if "新生儿" in text:
        return "0"

    if "半岁" in text:
        return "0"

    age_text = re.findall(".*岁", text)
    if age_text:
        age_xx = changeChineseNumToArab(age_text[0])
        age_xxx = re.findall("(\d+)岁", age_xx)
        if len(age_xxx) > 0:
            return age_xxx[0]

    age_text = re.findall("(\d+)岁", text)
    if age_text:
        return age_text[0]

    age_text = re.findall(".*周", text)
    if age_text:
        age = changeChineseNumToArab(age_text[0])
        if len(re.findall("(\d+)周", age)) > 0:
            return "0"

    age_text = re.findall("(\d+)周", text)
    if age_text:
        return "0"

    return ""

    # age_text = re.findall(".*", text)


def parse_content(a_json):
    a_dict = json.loads(a_json)
    question_title = a_dict.get("question_title")
    question_detail = a_dict.get("question_detail")
    ask_selected = a_dict.get("ask_selected")

    question_title_soup = BeautifulSoup(question_title, "html5lib")
    question_detail_soup = BeautifulSoup(question_detail, "html5lib")
    ask_selected_soup = BeautifulSoup(ask_selected, "html5lib")

    title_keys = [_.get_text().strip(u"：") for _ in question_title_soup.find_all("a")]
    title_keys = title_keys[1:] if len(title_keys) > 1 else title_keys
    disease_maybe = [_ for _ in title_keys if _ not in all_department and not _.endswith("科")]
    disease_maybe = list(set(disease_maybe))
    big_department = title_keys[0] if len(title_keys) >= 1 else ""
    department_keys = [_ for _ in title_keys if _ in all_department]

    if not department_keys:
        return {}

    ask_tit = question_detail_soup.find_all("p", attrs={"class": "ask_tit"})
    ask_tit_txt = u"。".join([re.sub("\s", "", _.get_text()) for _ in ask_tit])

    mation = question_detail_soup.find_all("p", attrs={"class": "mation"})
    mation_txt = u"。".join([re.sub("\s", "", _.get_text()) for _ in mation])

    ask_hid = question_detail_soup.find_all("div", attrs={"class": "ask_hid"})
    ask_hid_txt = u"。".join([re.sub("\s", "", _.get_text()) for _ in ask_hid])

    txt_label_tag = question_detail_soup.find("p", attrs={"class": "txt_label"})
    txt_label_a_tag = txt_label_tag.find_all("a") if txt_label_tag else []
    txt_label_s = [_.get_text() for _ in txt_label_a_tag] if txt_label_a_tag else []

    return {
        "mation": mation_txt,
        "ask_hid_txt": ask_hid_txt,
        "department": department_keys[-1],
        "big_department": big_department,
        "sex": parse_sex(mation_txt) or parse_sex(str(question_detail_soup)),
        "label": json.dumps(txt_label_s, ensure_ascii=False),
        "age": parse_age(mation_txt) or parse_age(str(question_detail_soup)),
        "disease": json.dumps(disease_maybe, ensure_ascii=False),
    }


def clean_data():
    for a_f_name in ask_page_list:
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


