# encoding: utf-8
import codecs
import csv
import os

f_path = "/Users/hy/Downloads/medical_data_set/3ddd.csv"

this_file_path = os.path.dirname(__file__)
prj_path = os.path.join(this_file_path, os.pardir)
train_data_dir = os.path.join(prj_path, "data/train_data2")

fp = codecs.open(f_path)
reader = csv.reader(fp)
rows = [row for row in reader]
for row in rows:
    (office_type, title, gender, age, label, office, disease, content) = row


# while True:
#     a_line = fp.readline()
#     if not a_line:
#         break
#
#     print(a_line)
