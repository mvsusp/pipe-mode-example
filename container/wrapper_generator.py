#!/usr/bin/env python
import random
from argparse import ArgumentParser
from os import path
from subprocess import check_call, check_output

import sagemaker
from sagemaker.estimator import Estimator
from tensorflow.python.data import Dataset

import sagemaker_containers
from imagenet_main import parse_record
from sagemaker_containers.tf_record_generator import TFRecordDatasetGenerator
from sagemaker_containers.tf_record_stream_reader import TFRecordStreamReader
import tensorflow as tf

if __name__ == '__main__':

    def random_generator():
        while True:
            yield str(random.randint(1, 100))


    r_generator = random_generator()
    generator = TFRecordStreamReader('training')()

    def wrapper_generator():
        for i in xrange(5):
            yield next(r_generator)


    dataset = Dataset.from_generator(wrapper_generator, tf.string, [])
    # dataset = dataset.map(lambda value: parse_record(value, True))

    sess = tf.Session()
    iterator = dataset.make_initializable_iterator()

    sess.run(iterator.initializer)

    while True:
        print(sess.run(iterator.get_next()))
        print(iterator.get_next())
