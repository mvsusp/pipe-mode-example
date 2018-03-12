#!/usr/bin/env python

from tensorflow.python.data import Dataset
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.ops.metrics_impl import mean_absolute_error
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.python.training.training_util import get_global_step

import resnet
from imagenet_main import parse_record, ImagenetModel
from sagemaker_containers.tf_record_stream_reader import TFRecordStreamReader


def imagenet_model_fn(features, labels, mode):
    """Our model_fn for ResNet to be used with our Estimator."""
    learning_rate_fn = resnet.learning_rate_with_decay(
        batch_size=32, batch_denom=256,
        num_images=1281167, boundary_epochs=[30, 60, 80, 90],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])

    return resnet.resnet_model_fn(features, labels, mode, ImagenetModel,
                                  resnet_size=50,
                                  weight_decay=1e-4,
                                  learning_rate_fn=learning_rate_fn,
                                  momentum=0.9,
                                  data_format=None,
                                  loss_filter_fn=None,
                                  multi_gpu=False)


if __name__ == '__main__':
    import tensorflow as tf

    print(tf.__version__)


    def input_fn():
        gen = TFRecordStreamReader('training')

        dataset = Dataset.from_generator(gen, tf.string)
        dataset = dataset.map(lambda value: parse_record(value, True), num_parallel_calls=5)
        dataset = dataset.prefetch(32)
        dataset = dataset.repeat(2)
        dataset = dataset.batch(32)
        return dataset


    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    # Set up a RunConfig to save checkpoint and set session config.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                                  session_config=session_config)

    estimator = tf.estimator.Estimator(imagenet_model_fn, config=run_config)

    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)

    estimator.train(input_fn, max_steps=1000000000, hooks=[logging_hook])
