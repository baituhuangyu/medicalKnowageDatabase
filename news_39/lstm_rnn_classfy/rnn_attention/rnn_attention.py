# encoding: utf-8


import tensorflow as tf
import numpy as np
import random
import copy
import os
import json
import random

import sys

this_file_path = os.path.dirname(__file__)


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

        self.batch_size = 16

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
            yield np.array(content_idx).reshape(-1, 1, 400), label_inx

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


class RNNAttentionModel(object):
    def __init__(self,
                vocab_size,
                embedding_size,
                word_num_hidden,
                word_attention_size,
                sentence_num_hidden,
                sentence_attention_size,
                num_classes,
                learning_rate,
                epoch,
                ):

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.word_num_hidden = word_num_hidden
        self.word_attention_size = word_attention_size
        self.sentence_num_hidden = sentence_num_hidden
        self.sentence_attention_size = sentence_attention_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.checkpointDir = "model/rnn_attention/"

        self.sess = tf.Session()
        self._placeholder_layers()
        self._embedding_layers()
        self._word_encoder_layers()
        self._word_attention_layers()
        self._sentence_encoder_layers()
        self._sentence_attention_layers()
        self._inference()
        self._build_train_op()

    def _placeholder_layers(self):
        # batch * sentence * words
        # 这里的 sentence 为 1，因为只有一个句子。
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="inputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None], name="targets")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="keep_prob")

        self.word_length = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(0, self.inputs.dtype), self.inputs), tf.int32), axis=-1
        )
        self.sentence_length = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(0, self.inputs.dtype), self.word_length), tf.int32), axis=-1
        )

    def _embedding_layers(self):
        with tf.variable_scope(name_or_scope="embedding_layers"):
            embedding_matrix = tf.get_variable(
                name="embedding_matrix", shape=[self.vocab_size, self.embedding_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )
            self.embedded_inputs = tf.nn.embedding_lookup(params=embedding_matrix, ids=self.inputs)
            print(self.embedded_inputs.shape)
            # [B * S * W * D]
            self.origin_shape = tf.shape(self.embedded_inputs)
            self.origin_shape_b, self.origin_shape_s, self.origin_shape_w, self.origin_shape_d = \
                self.origin_shape[0], self.origin_shape[1], self.origin_shape[2], self.origin_shape[3]

    def _word_encoder_layers(self):
        # 先做单个句子。
        with tf.variable_scope(name_or_scope="word_encoder_layers"):
            cell_fw = tf.nn.rnn_cell.GRUCell(num_units=self.word_num_hidden)
            cell_bw = tf.nn.rnn_cell.GRUCell(num_units=self.word_num_hidden)
            word_inputs = tf.reshape(self.embedded_inputs, [self.origin_shape_b * self.origin_shape_s,
                                                            self.origin_shape_w, self.embedding_size])
            word_length = tf.reshape(self.word_length, [self.origin_shape_b * self.origin_shape_s])
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=word_inputs, sequence_length=word_length,
                dtype=tf.float32, time_major=False
            )
            # 连起来
            self.word_encoder_output = tf.nn.dropout(x=tf.concat([output_fw, output_bw], axis=-1), keep_prob=self.keep_prob)

    def _word_attention_layers(self):
        with tf.variable_scope("word_attention_layers"):
            w_w = tf.get_variable(
                name="w_w", shape=[2 * self.word_num_hidden, self.word_attention_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            b_w = tf.get_variable(name="b_w", shape=[self.word_attention_size], initializer=tf.constant_initializer(0.))
            u_w = tf.get_variable(
                name="u_w", shape=[self.word_attention_size, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))

            v = tf.tanh(tf.nn.xw_plus_b(tf.reshape(self.word_encoder_output, [-1, 2 * self.word_num_hidden]), w_w, b_w))
            # 第3维的1是补的，为了能进行乘法运算，alpha第四维 * self.word_encoder_output 的第3维，结果见matmul
            alpha = tf.nn.softmax(tf.reshape(tf.matmul(v, u_w), [self.origin_shape_b*self.origin_shape_s, 1, self.origin_shape_w]))
            # si
            self.word_attention_output = tf.reduce_sum(tf.matmul(alpha, self.word_encoder_output), axis=1)

    def _sentence_encoder_layers(self):
        with tf.variable_scope(name_or_scope="sentence_encoder_layers"):
            cell_fw = tf.nn.rnn_cell.GRUCell(num_units=self.sentence_num_hidden)
            cell_bw = tf.nn.rnn_cell.GRUCell(num_units=self.sentence_num_hidden)

            sentence_level_inputs = tf.reshape(self.word_attention_output, [
                self.origin_shape_b, self.origin_shape_s, 2 * self.word_num_hidden])

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=sentence_level_inputs,
                sequence_length=self.sentence_length,
                dtype=tf.float32, time_major=False
            )

            self.sentence_encoder_output = tf.nn.dropout(x=tf.concat([output_fw, output_bw], axis=-1),
                                                         keep_prob=self.keep_prob)

    def _sentence_attention_layers(self):
        with tf.variable_scope("sentence_attention_layers"):
            w_1 = tf.get_variable(
                name="w_1", shape=[2 * self.sentence_num_hidden, self.sentence_attention_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            b_1 = tf.get_variable(name="b_1", shape=[self.sentence_attention_size], initializer=tf.constant_initializer(0.))
            u = tf.get_variable(
                name="w_2", shape=[self.sentence_attention_size, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            v = tf.nn.xw_plus_b(tf.reshape(self.sentence_encoder_output, [-1, 2 * self.sentence_num_hidden]), w_1, b_1)  # B*T*A
            s = tf.matmul(v, u)
            alphas = tf.nn.softmax(tf.reshape(s, [self.origin_shape[0], 1, self.origin_shape[1]]))
            self.sentence_attention_output = tf.reduce_sum(tf.matmul(alphas, self.sentence_encoder_output), axis=1)

    def _inference(self):
        with tf.variable_scope("train_op"):
            w = tf.get_variable(
                name="w", shape=[2 * self.sentence_num_hidden, self.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(
                name="b", shape=[self.num_classes], initializer=tf.constant_initializer(0.)
            )
            self.logits = tf.matmul(self.sentence_attention_output, w) + b
            self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.targets)
            self.accuracy_val = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

    def _build_train_op(self):
        self.total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits)
        self.loss = tf.reduce_mean(self.total_loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

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
        for i in range(self.epoch):
            dataPreClass.reset()
            for input_x, input_y in dataPreClass:
                step += len(input_y)
                _iter += 1
                _, loss, acc = self.sess.run(
                    fetches=[self.train_op, self.loss, self.accuracy_val],
                    feed_dict={self.inputs: input_x, self.targets: input_y, self.keep_prob: 0.5})
                print("<Train>\t Epoch: [%d] Iter: [%d] Step: [%d] Loss: [%.3F]\t Acc: [%.3f]" %
                      (i+1, _iter, step, loss, acc))
            self._save()

    def test(self):
        print("\nbegin test ....\n")
        _iter = 0
        dataPreClass = DataPreClass(model="test")
        for input_x, input_y in dataPreClass:
            _iter += 1
            acc, loss = self.sess.run(
                fetches=[self.accuracy_val, self.loss],
                feed_dict={self.inputs: input_x, self.targets: input_y, self.keep_prob: 1.})
            print("<Test>\t Iter: [%d] Loss: [%.3F]\t Acc: [%.3f]" %
                  (_iter, loss, acc))


if __name__ == '__main__':
    #sequence_length=400, num_classes=45, vocab_size=6020,
    #                     embedding_size=128, filter_sizes=[3, 6, 9], num_filters=256)
    rnn_attention_model = RNNAttentionModel(
        vocab_size=6020,
        embedding_size=128,
        word_num_hidden=64,
        word_attention_size=64,
        sentence_num_hidden=64,
        sentence_attention_size=32,
        num_classes=45,
        learning_rate=1e-3,
        epoch=3,
    )

    rnn_attention_model.train()
    rnn_attention_model.test()












