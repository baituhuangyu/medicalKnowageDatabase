#!/usr/bin/env python3
# encoding: utf-8

import tensorflow as tf
import numpy as np
import random
import copy
import os
import json
import sys
import random

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

this_file_path = os.path.dirname(__file__)


def make_train_test_data():
    all_data_fp = os.path.join(this_file_path, '../../data/all_data/all_data.txt')
    train_data_fp = os.path.join(this_file_path, '../../data/all_data/train_data.txt')
    test_data_fp = os.path.join(this_file_path, '../../data/all_data/test_data.txt')

    train_data_f = open(train_data_fp, "w+")
    test_data_f = open(test_data_fp, "w+")

    with open(all_data_fp, "r") as fp:
        # print("open dataset")
        lines = fp.readlines()
        for line in lines:
            if random.random() * 10 > 1:
                # train
                train_data_f.write(line)
            else:
                test_data_f.write(line)


class DataPreClass(object):
    # train_data_fp = "data/trainset.txt"
    # test_data_fp = "data/testset.txt"
    # chinese_vocab_fp = "data/chinese_vocab.txt"

    # train_data_fp = os.path.join(this_file_path, '../../data/all_data/all_data.txt')
    # test_data_fp = os.path.join(this_file_path, '../../data/all_data/all_data.txt')

    train_data_fp = os.path.join(this_file_path, '../../data/all_data/train_data.txt')
    test_data_fp = os.path.join(this_file_path, '../../data/all_data/test_data.txt')

    chinese_vocab_fp = os.path.join(this_file_path, '../../data/all_data/chinese_vocab.txt')

    def __init__(self, model="train"):
        if model == "train":
            self.file_path = self.train_data_fp
        else:
            self.file_path = self.test_data_fp

        self.batch_size = 1

        labels = ["不孕不育", "中医内科", "中医妇科", "中医科", "中医骨伤", "乳腺外科", "产科", "其他传染病", "内分泌科", "减肥", "口腔科", "呼吸内科", "妇科", "寄生虫科", "小儿内科", "小儿外科", "小儿精神科", "心胸外科", "心血管内科", "性病科", "整形美容", "新生儿科", "普外科", "泌尿外科", "消化内科", "烧伤科", "男科", "皮肤科", "眼科", "神经内科", "神经外科", "精神心理科", "结核科", "耳鼻喉科", "肛肠外科", "肝病科", "肝胆外科", "肾内科", "肿瘤科", "胃肠外科", "血液科", "血管外科", "针灸推拿", "风湿免疫科", "骨科"]

        self.labels_map = dict([(str(v), idx) for idx, v in enumerate(labels)])
        self.vocab_map = self.read_vocab_map()
        self.dataset = self._read_file(self.file_path)
        self.iterobj = self.reset()

        self.iter_data = self.iter_data_set().__iter__()

    def _read_file(self, file_path):
        with open(file_path, "r") as fp:
            # print("open dataset")
            dataset = fp.readlines()
            random.shuffle(dataset)
            used_line_list = []
            for line in dataset:
                line = line.strip()
                if not line:
                    continue
                a_dict = json.loads(line)
                label = a_dict.get("department", "")
                content = a_dict.get("ask_hid_txt", "")
                used_line_list.append({"label": label, "content": content})
            self.dataset = used_line_list
            return self.dataset

    def iter_data_set(self):
        for a_line in self.dataset:
            if not a_line:
                continue
            # label, _content = a_line.split("\t")
            label, _content = a_line.get("label", ""),  a_line.get("content", "")
            if not label or not _content:
                continue

            if len(_content) >= 400:
                content = _content[:400]
            else:
                content = _content + " " * (400 - len(_content))
            content_idx = [self.vocab_map.get(str(k), 0) for k in content]
            label_inx = self.labels_map[str(label)]
            yield np.array(content_idx).reshape(-1, 400), label_inx

    def reset(self):
        dataset = [_ for _ in copy.deepcopy(self.dataset)]
        random.shuffle(dataset)
        self.dataset = dataset
        self.iter_data = self.iter_data_set().__iter__()
        return self.dataset

    def read_vocab_map(self):
        with open(self.chinese_vocab_fp, "r") as fp:
            print("open read_vocab_map")
            vocab_list = fp.readlines()
            vocab_list = [_.strip() for _ in vocab_list]

        return dict([(str(v), idx) for idx, v in enumerate(vocab_list)])

    def __next__(self):
        document_lst = []
        deal_x = []
        deal_y = []
        count = 0
        try:
            while count < self.batch_size:
                cur = next(self.iter_data)
                if not cur:
                    continue
                count += 1
                document_lst.append(cur)
                deal_x.append(cur[0][0])
                # deal_y.append(cur[1][0])
                deal_y.append(cur[1])
        except StopIteration as iter_exception:
            if count == 0:
                raise iter_exception

        return np.array(deal_x, dtype=np.int32), np.array(deal_y, dtype=np.int32)

    def __iter__(self):
        return self


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters):

        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.checkpointDir = "model/cnn/"

        self._define()
        self._embedding_layer()
        self._conv()
        self._dropout()
        self._prediction()
        self._cal_loss()
        self._cal_acc()
        self._grad()

        self.sess = tf.Session()

    def _define(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _embedding_layer(self):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding_matrix = tf.get_variable(
                name="embedding_matrix", shape=[self.vocab_size, self.embedding_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )
            self.embedded_chars = tf.nn.embedding_lookup(embedding_matrix, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, 1)
            print("self.embedded_chars_expanded.shape ", self.embedded_chars_expanded.shape)

    def _conv(self):
        pooled_outputs = []
        for filter_size in self.filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % str(filter_size)):
                # Convolution Layer
                # filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                filter_shape = [1, filter_size, self.embedding_size, self.num_filters]
                kernel = tf.get_variable(shape=filter_shape, name="conv-maxpool-%s-W" % str(filter_size), initializer=tf.truncated_normal_initializer(stddev=0.01))
                bias = tf.get_variable(initializer=tf.constant_initializer(), shape=[self.num_filters], name="conv-maxpool-%s-b" % str(filter_size))
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    kernel,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 1, self.sequence_length - filter_size + 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

    def _dropout(self):
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

    def _prediction(self):
        with tf.name_scope("output"):
            W = tf.get_variable(shape=[self.num_filters_total, self.num_classes], initializer=tf.truncated_normal_initializer(stddev=0.1), name="W")
            # b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            b = tf.get_variable(name="b", shape=[self.num_classes], dtype=tf.float32)
            # self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.scores = tf.matmul(self.h_drop, W, name="logits") + b

            self.predictions = tf.argmax(self.scores, 1, name="predictions")

    def _cal_loss(self):
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)

            self.loss = tf.reduce_mean(losses)

    def _cal_acc(self):
        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            # correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def _grad(self):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    def _save(self):
        if not tf.gfile.Exists(self.checkpointDir):
            tf.gfile.MakeDirs(self.checkpointDir)
        saver = tf.train.Saver()
        saver.save(sess=self.sess, save_path=self.checkpointDir + "model")

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        print("\nbegin train ....\n")
        step = 0
        _iter = 0
        dataPreClass = DataPreClass(model="train")
        for i in range(3):
            dataPreClass.reset()
            while True:
            # for input_x, input_y in dataPreClass:
                try:
                    input_x, input_y = dataPreClass.__next__()
                    # print(input_x)
                    # print(input_y)
                    _iter += 1
                    step += len(input_x)
                    _, loss, acc = self.sess.run(
                        fetches=[self.train_op, self.loss, self.accuracy],
                        feed_dict={self.input_x: input_x, self.input_y: input_y, self.dropout_keep_prob: 0.5})
                    print("<Train>\t Epoch: [%d] Iter: [%d] Step: [%d] Loss: [%.3F]\t Acc: [%.3f]" %
                          (i+1, _iter, step, loss, acc))
                except StopIteration:
                    break
            self._save()

    def test(self):
        print("\nbegin test ....\n")
        testset = DataPreClass("test")
        while True:
        # for input_x, input_y in testset:
            try:
                input_x, input_y = testset.__next__()
                acc, loss = self.sess.run(
                    fetches=[self.accuracy, self.loss],
                    feed_dict={self.input_x: input_x, self.input_y: input_y, self.dropout_keep_prob: 1.})
                print("Loss: [%.3F]\t Acc: [%.3f]" % (loss, acc))
            except StopIteration:
                break


class PredictModel(object):
    def __init__(self):
        self.checkpointDir = "model/cnn/"
        self.chinese_vocab_fp = os.path.join(this_file_path, '../../data/all_data/chinese_vocab.txt')
        labels = ["不孕不育", "中医内科", "中医妇科", "中医科", "中医骨伤", "乳腺外科", "产科", "其他传染病", "内分泌科", "减肥", "口腔科", "呼吸内科", "妇科",
                  "寄生虫科", "小儿内科", "小儿外科", "小儿精神科", "心胸外科", "心血管内科", "性病科", "整形美容", "新生儿科", "普外科", "泌尿外科", "消化内科",
                  "烧伤科", "男科", "皮肤科", "眼科", "神经内科", "神经外科", "精神心理科", "结核科", "耳鼻喉科", "肛肠外科", "肝病科", "肝胆外科", "肾内科",
                  "肿瘤科", "胃肠外科", "血液科", "血管外科", "针灸推拿", "风湿免疫科", "骨科"]

        self.labels_map = dict([(idx, str(v)) for idx, v in enumerate(labels)])
        self.vocab_map = self.read_vocab_map()

    def read_vocab_map(self):
        with open(self.chinese_vocab_fp, "r") as fp:
            print("open read_vocab_map")
            vocab_list = fp.readlines()
            vocab_list = [_.strip() for _ in vocab_list]

        return dict([(str(v), idx) for idx, v in enumerate(vocab_list)])

    def _encode_content(self, content_list):
        encode_content_list = []
        for sentence in content_list:
            if len(sentence) >= 400:
                sentence = sentence[:400]
            else:
                sentence = sentence + " " * (400 - len(sentence))
            content_idx = [self.vocab_map.get(str(k), 0) for k in sentence]
            encode_content_list.append(np.array(content_idx).reshape(-1, 400))

        return np.array(encode_content_list, dtype=np.int32).reshape(-1, 400)

    def __cnn_by_meta_graph(self):
        checkpoint_file = tf.train.latest_checkpoint(self.checkpointDir)
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]

    def cnn_predict_by_meta_graph(self, content_list):
        input_batch = self._encode_content(content_list)
        self.__cnn_by_meta_graph()
        batch_predictions = self.sess.run(self.predictions, {self.input_x: input_batch, self.dropout_keep_prob: 1.0})
        return [self.labels_map[_] for _ in batch_predictions]


if __name__ == '__main__':
    # make_train_test_data()
    # exit(0)

    # cnn_model = TextCNN(sequence_length=400, num_classes=45, vocab_size=6020,
    #                     embedding_size=128, filter_sizes=[3, 6, 9], num_filters=256)
    #
    # cnn_model.train()
    # cnn_model.test()
    content = [
        "双肾体积偏小,大小分别约为8.67×2.68cm(右），8.81×3.71（左），形态尚正常，左肾中部实质可见一强回声斑，大小约为0.33×0.23cm，右肾实质及双肾窦回声正常。",
    ]

    pm = PredictModel()
    predict_result = pm.cnn_predict_by_meta_graph(content)
    print(json.dumps(predict_result, indent=4, ensure_ascii=False))
