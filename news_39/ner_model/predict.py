import os
import json
import codecs
import tensorflow as tf

saved_model_dir = os.path.join(os.path.abspath(__file__), os.pardir, "ner_model")


class PredictNer(object):
    def __init__(self, saved_model, source_data):
        graph = tf.Graph()
        self.saved_model = saved_model
        self.chinese_vocab = self.__load_chinese_vocab()
        self.source_data = source_data
        self.__pre_handle_sentence()
        self.sess = tf.Session(graph=graph)
        self.__get_tensor_name()

    def __get_tensor_name(self):
        meta_graph_def = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], self.saved_model)
        signature = meta_graph_def.signature_def
        self.input_x = self.sess.graph.get_tensor_by_name(signature["ner_name"].inputs["inputs_x"].name)
        self.keep_prob = self.sess.graph.get_tensor_by_name(signature["ner_name"].inputs["keep_prob"].name)
        self.decode_tags = self.sess.graph.get_tensor_by_name(signature["ner_name"].outputs["decode_tags"].name)

    @staticmethod
    def __load_chinese_vocab():
        cv = dict()
        with codecs.open(os.path.join(os.path.dirname(__file__), os.pardir, "data/all_data/chinese_vocab.txt"), "r", "utf8") as f:
            for i, line in enumerate(f.readlines()):
                cv[line.strip()] = i
        return cv

    def __pre_handle_sentence(self):
        input_x = []
        for _text in self.source_data:
            sentence_lst = []
            for s in _text:
                _id = self.chinese_vocab.get(s, -1)
                if _id == -1:
                    raise Exception("count of chinese vocab not enough")
                sentence_lst.append(_id)
            input_x.append(sentence_lst)
        max_len = max([len(x) for x in input_x])
        self.content = tf.keras.preprocessing.sequence.pad_sequences(
            input_x, maxlen=max_len, padding="post", truncating="post", dtype="int32", value=0)

    def batch_predict_ner(self):
        decode_tags = self.sess.run(self.decode_tags, feed_dict={self.input_x: self.content, self.keep_prob: 1.})
        result_ner = []
        for i in range(len(decode_tags)):
            sentence_ner = []
            word_ner = []
            for j in range(len(decode_tags[i])):
                if decode_tags[i].tolist()[j] != 0:
                    word_ner.append(self.source_data[i][j])
                elif decode_tags[i].tolist()[j-1] == 3:
                    if len(word_ner) != 0:
                        sentence_ner.append("".join(word_ner))
                        word_ner = []
            else:
                result_ner.append(sentence_ner)
        return result_ner


if __name__ == "__main__":
    content = [
        "隐匿性肾炎，你好医生，尿蛋白正常，隐血+号，红细胞20几个，这种病补充提问：是没法医的吗补充提问：有胡桃夹",
        "昨天下午开始感觉喉咙有点干，有点痛，今天还是一样，傍晚开始，全身抹起来都痛，除了四肢。早几天早上被尿憋醒，尿了很久，尿完感觉阴道有点微痛，之后几天尿完都会有点微痛的症状，尿有点黄，是尿毒症吗，还是感冒了",
    ]
    ner = PredictNer("ner_model", content)
    pred = ner.batch_predict_ner()
    print(json.dumps(pred, indent=2, ensure_ascii=False))
