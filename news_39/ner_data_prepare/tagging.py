#!/usr/bin/env python
# encoding: utf-8

import os
import json
import re
import codecs
import sys
from collections import Counter

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

this_file_path = os.path.dirname(__file__)
prj_path = os.path.join(this_file_path, os.pardir)
clean_data_path = os.path.join(prj_path, "data/all_data/all_data.txt")
tagged_data_path = os.path.join(prj_path, "data/all_data/all_data_tagged.txt")
chinese_vocab_data_path = os.path.join(prj_path, "data/all_data/chinese_vocab.txt")


def tagging_task():
    tagged_data_fs = codecs.open(tagged_data_path, "w+")
    with codecs.open(clean_data_path, "r") as fp:
        while True:
            a_line = fp.readline()
            if not a_line:
                break

            sentence_tagged = tagging_row(a_line.strip())
            if not sentence_tagged:
                continue

            tagged_txt = "\n".join([" ".join(_) for _ in sentence_tagged])

            tagged_data_fs.write(tagged_txt+"\n"+"end"+"\n")

    tagged_data_fs.close()


def tagging_row(row):
    row_json = json.loads(row)
    label_list = json.loads(row_json["label"])
    label_list = sorted(label_list, key=len, reverse=True)
    a_sentence = row_json["ask_hid_txt"]

    label_list_in_sentence = [_ for _ in label_list if _ in a_sentence]
    if not label_list_in_sentence:
        return []

    sentence_tagged = [[_, "O"] for _ in a_sentence]
    sentence_len = len(sentence_tagged)
    for index, w_k in enumerate(sentence_tagged):
        for a_label in label_list_in_sentence:
            if not a_label:
                continue

            if len(a_label) + index > sentence_len:
                continue

            next_may_be = sentence_tagged[index:index + len(a_label)]

            if any([True for _ in next_may_be if _[1] != "O"]):
                continue

            if a_label == "".join([_[0] for _ in next_may_be]):
                # tagging
                next_may_be[0][1] = "B-KEY"
                next_may_be[-1][1] = "E-KEY"
                if len(next_may_be) > 2:
                    for xxx in next_may_be[1: -1]:
                        xxx[1] = "I-KEY"

    return sentence_tagged


def build_vocab():
    vocab_size = 50000
    all_lines = []

    with codecs.open(tagged_data_path, "r", "utf-8") as fp:
        lines = fp.readlines()
        for line in lines:
            ws = line.strip().split(" ")
            if ws and ws != "end":
                all_lines.append(ws[0])

    # all_data = []
    # for content in lines:
    # #     all_data.extend(content)

    all_data = all_lines

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    words_str = '\n'.join(words) + '\n'
    codecs.open(chinese_vocab_data_path, 'w', "utf-8").write(words_str)


if __name__ == '__main__':
    # tagging_task()

    build_vocab()


