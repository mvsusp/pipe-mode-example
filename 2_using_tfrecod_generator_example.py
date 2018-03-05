#!/usr/bin/env python
import os
import logging
from tf_record_generator import TFRecordDatasetGenerator

logging.basicConfig()
logging.getLogger('tf_record_generator').setLevel(logging.DEBUG)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow.python.data import Dataset

    generator = TFRecordDatasetGenerator('data/fake_stream_for_generator_example', 21, CURRENT_DIR)
    dataset = Dataset.from_generator(generator, tf.string)
    iterator = dataset.make_one_shot_iterator()

    sess = tf.Session()
    tf_serialized_example = iterator.get_next()

    for i in xrange(42):
        feature_map = {'image/filename': tf.FixedLenFeature([], dtype=tf.string),
                       'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64)}

        example_tensor = tf.parse_single_example(tf_serialized_example, feature_map)

        tf_example_map = sess.run(example_tensor)

        image_filename, label = tf_example_map['image/filename'], tf_example_map['image/class/label']

        print('TF examples contains image {} and label {}'.format(image_filename, label))

        tf_serialized_example = iterator.get_next()
