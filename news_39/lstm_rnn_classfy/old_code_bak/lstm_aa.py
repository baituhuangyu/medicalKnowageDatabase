# encoding: utf-8

import os
import json
from collections import Counter
import codecs
import pdb

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr

import csv

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

this_file_path = os.path.dirname(__file__)
prj_path = os.path.join(this_file_path, os.pardir)
clean_data_dir = os.path.join(prj_path, "data/clean_data")
train_data_dir = os.path.join(prj_path, "data/train_data")

clean_data_list = os.listdir(clean_data_dir)


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return str(word)
    else:
        return word


def native_content(content):
    if not is_py3:
        return str(content)
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    return codecs.open(filename, mode, "utf-8")


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    # contents.append(list(native_content(content)))
                    content_list = json.loads(content)
                    if len(content_list) > 2:
                        contents.append(content_list[0:2] + list(set(content_list[2:])))
                        labels.append(native_content(label))
            except:
                pass
    return contents, labels


def read_vocab():
    """读取词汇表"""
    vocab_dir = os.path.join(train_data_dir, "vocab.txt")
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def read_category():
    """读取分类目录，固定"""
    categories = ["不孕不育", "中医内科", "中医妇科", "中医科", "中医骨伤", "乳腺外科", "产科", "其他传染病", "内分泌科", "减肥", "口腔科", "呼吸内科", "妇科", "寄生虫科", "小儿内科", "小儿外科", "小儿精神科", "心胸外科", "心血管内科", "性病科", "整形美容", "新生儿科", "普外科", "泌尿外科", "消化内科", "烧伤科", "男科", "皮肤科", "眼科", "神经内科", "神经外科", "精神心理科", "结核科", "耳鼻喉科", "肛肠外科", "肝病科", "肝胆外科", "肾内科", "肿瘤科", "胃肠外科", "血液科", "血管外科", "针灸推拿", "风湿免疫科", "骨科"]

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def gen_model_data():
    all_new_line = []
    line_set = set()
    department_set = set()
    train_data_path = os.path.join(train_data_dir, "train_data.txt")
    val_data_path = os.path.join(train_data_dir, "val.txt")
    fp2 = codecs.open(train_data_path, "w+", "utf-8")
    val_fp2 = codecs.open(val_data_path, "w+", "utf-8")

    max_a_class_line = 10000000
    max_a_val_line = 100
    for a_f_name in clean_data_list:
        # print(a_f_name)
        a_count = 0
        a_val_count = 0
        a_f_name_path = os.path.join(clean_data_dir, a_f_name)

        with codecs.open(a_f_name_path, "r", "utf-8") as fp:
            all_lines = fp.readlines()
            for line in all_lines:
                if a_count > max_a_class_line:
                    break
                a_dict = json.loads(line.strip())

                department = a_dict["department"]
                department_set.add(department)

                # mation = a_dict["mation"]
                # ask_hid_txt = a_dict["ask_hid_txt"]

                # a_new_line_content = (mation + "。" + ask_hid_txt).replace("\t", "")
                # new_line = department + "\t" + a_new_line_content + "\n"
                # new_line = str(new_line)

                sex = a_dict["sex"]
                age = a_dict["age"]
                label = json.loads(a_dict["label"])
                a_new_line_content = json.dumps([sex, age]+label, ensure_ascii=False)
                new_line = department + "\t" + a_new_line_content + "\n"
                if a_new_line_content not in line_set:
                    # pdb.set_trace()
                    # all_new_line.append(new_line)
                    # print(str(new_line.encode("utf-8")))
                    try:
                        fp2.write(new_line)
                    except Exception as e:
                        print(e)
                    a_count += 1

                    if a_val_count <= max_a_val_line:
                        a_val_count += 1
                        val_fp2.write(new_line)

                line_set.add(a_new_line_content)

    fp2.close()

    print(json.dumps(sorted(list(department_set)), ensure_ascii=False))


def gen_vocab():
    vocab_size = 50000000
    train_data_path = os.path.join(train_data_dir, "train_data.txt")
    vocab_path = os.path.join(train_data_dir, "vocab.txt")
    all_lines = []

    with codecs.open(train_data_path, "r", "utf-8") as fp:
        lines = fp.readlines()
        # reader = csv.reader(fp)
        # rows = [row for row in reader]
        for line in lines:
            all_lines.extend(json.loads(line.split("\t")[1]))


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
    codecs.open(vocab_path, 'w', "utf-8").write(words_str)


if __name__ == '__main__':
    # gen_model_data()
    gen_vocab()
