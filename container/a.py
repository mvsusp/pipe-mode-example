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

    dataset = Dataset.from_tensor_slices( tf.zeros(shape=[20]))
    # dataset = Dataset.from_tensor_slices([1,2,3,4,5,6,7,8,9])

    gen = TFRecordStreamReader('training')()


    def al_gen():



    def getty():
        """ Iterator of TF serialized examples.

        As data is read from the stream, more data is streamed in from S3.
        For better performance, read the maximum possible in advance without crossing memory constraints.
        """

        counter = 1

        while True:
            yield random.randint(1, 100)
            if counter == 10:
                return
            counter += 1

    generator = getty()

    def next_record(t):
        return tf.cast(next(generator), dtype=tf.int32)

    def next_record(t):
        return tf.cast(random.randint(1, 100), dtype=tf.int32)


    def my_func():
        # x will be a numpy array with the contents of the placeholder below

        return tf.convert_to_tensor(random.randint(1, 100), dtype=tf.int32)


    inp = tf.placeholder(tf.int32)

    aaa = lambda x: tf.py_func(my_func, [], tf.int32)

    dataset = dataset.map(aaa)

    # records = []
    #
    # for i in xrange(20):
    #     records.append(next(gen))
    #
    # dataset = tf.data.Dataset.from_tensor_slices(records)

    # dataset = dataset.map(lambda value: parse_record(value, True))

    sess = tf.Session()
    iterator = dataset.make_initializable_iterator()

    sess.run(iterator.initializer)

    while True:
        sess.run(iterator.get_next())
        print(sess.run(iterator.get_next()))
        print(iterator.get_next())