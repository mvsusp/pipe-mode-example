#!/usr/bin/env python
import random
from argparse import ArgumentParser
from os import path, getpid
from subprocess import check_call, check_output

import psutil
import sagemaker
from sagemaker.estimator import Estimator
from tensorflow.python.data import Dataset

import sagemaker_containers
from imagenet_main import parse_record
from sagemaker_containers.tf_record_generator import TFRecordDatasetGenerator
from sagemaker_containers.tf_record_stream_reader import TFRecordStreamReader
import tensorflow as tf

if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf

    N = 10 * 1024 * 1024


    def generate():
        data = np.random.rand(N)
        for k in range(N):
            yield data[k].copy()


    graph = tf.Graph()
    with graph.as_default():
        x = tf.data.Dataset \
            .from_generator(generate, tf.float32) \
            .make_one_shot_iterator() \
            .get_next()

    process = psutil.Process(getpid())
    before = process.memory_percent()

    while True:
        session = tf.Session(graph=graph)
        session.run(x)  # <--- PUT A BREAKPOINT HERE!

        after = process.memory_percent()
        print("MEMORY CHANGE %.4f -> %.4f" % (before, after))
        before = after

        #  Be careful running the code without it!
        session.close()
