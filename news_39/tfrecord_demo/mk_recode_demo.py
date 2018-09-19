#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

import numpy as np
import pandas as pd

# make tfrecords
train_frame = pd.read_csv(filepath_or_buffer="./demo_dataset.csv")
train_labels_frame = train_frame.pop(item="label")

train_values = train_frame.values
train_size = train_values.shape[0]
train_labels_values = train_labels_frame.values

writer = tf.python_io.TFRecordWriter(path="train.tfrecords")
for i in range(train_size):
    image_raw = train_values[i].astype(np.float32).tobytes()
    label = train_labels_values[i].astype(np.float32).tobytes()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
            }
        )
    )
    writer.write(record=example.SerializeToString())

writer.close()

tfrecords = tf.python_io.tf_record_iterator("train.tfrecords")


# read tfrecords
for i in tfrecords:
    features = tf.parse_single_example(
        i,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)
    print(tf.Session().run(image))
    print(tf.Session().run(label))
