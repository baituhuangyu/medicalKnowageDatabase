# encoding: utf-8
import os
import codecs
import json


this_file_path = os.path.dirname(__file__)
prj_path = os.path.join(this_file_path, os.pardir)
train_data_dir = os.path.join(prj_path, "data/train_data")
network_data_dir = os.path.join(prj_path, "data_base/data")


def build_nodes():
    train_data_path = os.path.join(train_data_dir, "train_data.txt")
    department_path = os.path.join(network_data_dir, "department.txt")
    label_path = os.path.join(network_data_dir, "label.txt")
    department_label_path = os.path.join(network_data_dir, "department_label.txt")

    department_set = set()
    label_set = set()
    # department_label_set = set()
    department_label_dict = dict()

    with codecs.open(train_data_path, "r", "utf-8") as fp:
        lines = fp.readlines()

        for line in lines:
            a_lint_list = line.split("\t")
            a_department = a_lint_list[0]
            a_label_list = json.loads(a_lint_list[1])
            a_label_list = a_label_list[2:] if len(a_label_list) > 2 else []

            if a_department not in department_set:
                department_set.add(a_department)

            for a_label in a_label_list:
                a_label = a_label.replace(",", "")
                a_new_line = a_department + "," + a_label
                # if a_new_line not in department_label_dict:
                count_department_label = department_label_dict.get(a_new_line, 0)
                department_label_dict[a_new_line] = count_department_label + 1

                if a_label not in label_set:
                    label_set.add(a_label)

    with codecs.open(department_path, "w+") as department_path_fp:
        department_path_fp.write("\n".join(list(department_set)))

    with codecs.open(label_path, "w+") as label_path_fp:
        label_path_fp.write("\n".join(list(label_set)))

    with codecs.open(department_label_path, "w+") as department_label_path_fp:
        department_label_sorted = sorted(department_label_dict.items(), key=lambda _: _[1], reverse=True)
        department_label_path_fp.write("\n".join(["%s,%s" % _ for _ in department_label_sorted]))


build_nodes()
