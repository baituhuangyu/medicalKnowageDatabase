# encoding: utf-8

from ner_model.predict import PredictNer, saved_model_dir


def get_ner_content(content):
    ner = PredictNer(saved_model_dir, content)
    return ner.batch_predict_ner()

