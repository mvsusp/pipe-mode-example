#!/usr/bin/env python

from tensorflow.python.data import Dataset
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.ops.metrics_impl import mean_absolute_error
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.python.training.training_util import get_global_step

from imagenet_main import parse_record
from sagemaker_containers.tf_record_stream_reader import TFRecordStreamReader

if __name__ == '__main__':
    import tensorflow as tf

    print(tf.__version__)


    def input_fn():
        gen = TFRecordStreamReader('training')

        dataset = Dataset.from_generator(gen, tf.string)
        dataset = dataset.map(lambda value: parse_record(value, True), num_parallel_calls=5)
        return dataset.prefetch(32)


    def _model_fn(features, labels, mode):
        predictions = tf.layers.dense(
            features, 1, kernel_initializer=tf.zeros_initializer())

        loss = tf.losses.mean_squared_error(predictions, predictions)
        tf.summary.scalar('loss', loss)

        train_op = GradientDescentOptimizer(learning_rate=0.5).minimize(loss, get_global_step())
        eval_metric_ops = {
            'absolute_error': mean_absolute_error(predictions, predictions)
        }

        return EstimatorSpec(
            mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)


    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    # Set up a RunConfig to save checkpoint and set session config.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                                  session_config=session_config)

    estimator = tf.estimator.Estimator(_model_fn, config=run_config)

    tensors_to_log = {
        'loss': 'loss'
    }

    # profiler_hook = tf.train.ProfilerHook(save_steps=1000, output_dir=flags.model_dir)

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)

    estimator.train(input_fn, max_steps=1000000000, hooks=[logging_hook])
