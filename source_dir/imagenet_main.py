from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

import tensorflow as tf
from tensorflow.python.data import Dataset

import resnet
import vgg_preprocessing
from tf_record_generator import TFRecordDatasetGenerator

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

_NUM_IMAGES = 42 * 100000

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
logger.addHandler(_handler)

tf.logging.set_verbosity(logging.DEBUG)


def train_input_fn(training_dir, hyperpameters):
    """Preprocess and load training data.

    Returns:
        A dataset that can be used for iteration containing features and labels
    """

    # All the 100 tf records files from the training channel will stream in a file named training_0
    channel_name = 'training'
    return _input_fn(True, channel_name, hyperpameters['batch_size'])


def eval_input_fn(training_dir, hyperpameters):
    """Preprocess and load training data.

    Returns:
        A dataset that can be used for iteration containing features and labels
    """

    # All the 100 tf records files from the validation channel will stream in a file named validation_0
    channel_name = 'validation'
    return _input_fn(False, channel_name, hyperpameters['batch_size'])


def _input_fn(is_training, channel_name, batch_size, num_epochs=1, num_parallel_calls=1, multi_gpu=False):
    """Input function which provides batches for train or eval.
  Args:
    is_training: A boolean denoting whether the input is for training.
    channel_name: Channel that will for training/evaluation.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel. This can be optimized per data set but
        for generally homogeneous data sets, should be approximately the number of available CPU cores.
    multi_gpu: Whether this is run multi-GPU. Note that this is only required currently to handle the batch leftovers,
        and can be removed when that is handled directly by Estimator.

  Returns:
        A dataset that can be used for iteration containing features and labels
  """

    # Let's create a dataset generator that knows how to parse the tf records from the stream in a serialized TF example
    generator = TFRecordDatasetGenerator(channel_name)

    # We can create a tf.data.Dataset from the generator.
    dataset = Dataset.from_generator(generator, tf.string)

    # We prefetch a batch at a time, This can help smooth out the time taken to
    # load input files as we go through shuffling and processing.
    dataset = dataset.prefetch(batch_size)

    return resnet.process_record_dataset(dataset, is_training, batch_size,
                                         batch_size, _parse_record, num_epochs, num_parallel_calls,
                                         examples_per_epoch=_NUM_IMAGES, multi_gpu=multi_gpu)


def _parse_example_proto(tf_serialized_example):
    """Parses an Example proto containing a training example of an image.

  The dataset contains serialized Example protocol buffers. The Example proto is expected to contain features named
  image/encoded (a JPEG-encoded string) and image/class/label (int)

  Args:
    tf_serialized_example: scalar Tensor tf.string containing a serialized Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int64 containing the label.
  """
    feature_map = {'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
                   'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64)}

    tf_example_map = tf.parse_single_example(tf_serialized_example, feature_map)

    return tf_example_map['image/encoded'], tf_example_map['image/class/label']


def _parse_record(tf_serialized_example, is_training):
    """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed through preprocessing steps
  (cropping, flipping, and so on).

  Args:
    tf_serialized_example: scalar Tensor tf.string containing a serialized Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
"""
    image, label = _parse_example_proto(tf_serialized_example)

    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    # Results in a 3-D int8 Tensor. This will be converted to a float later,
    # during resizing.
    image = tf.image.decode_jpeg(image, channels=_NUM_CHANNELS)

    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        is_training=is_training)

    label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32)
    label = tf.one_hot(label, _NUM_CLASSES)

    return image, label


def model_fn(features, labels, mode, hyperparameters):
    """Our model_fn for ResNet to be used with our Estimator."""

    learning_rate_fn = resnet.learning_rate_with_decay(
        batch_size=hyperparameters['batch_size'], batch_denom=256, num_images=_NUM_IMAGES,
        boundary_epochs=[30, 60, 80, 90],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])

    return resnet.resnet_model_fn(
        features, labels, mode, ImagenetModel, resnet_size=hyperparameters['resnet_size'], weight_decay=1e-4,
        learning_rate_fn=learning_rate_fn, momentum=0.9, data_format=None, loss_filter_fn=None,
        multi_gpu=hyperparameters['multi_gpu'])


def serving_input_fn(hyperpameters):
    inputs = {'INPUTS': tf.placeholder(tf.float32, [None, 224, 224, 3])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


###############################################################################
# Creating the model
###############################################################################
class ImagenetModel(resnet.Model):

    def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES):
        """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
    """

        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            block_fn = resnet.building_block
            final_size = 512
        else:
            block_fn = resnet.bottleneck_block
            final_size = 2048

        super(ImagenetModel, self).__init__(
            resnet_size=resnet_size, num_classes=num_classes, num_filters=64, kernel_size=7, conv_stride=2,
            first_pool_size=3, first_pool_stride=2, second_pool_size=7, second_pool_stride=1, block_fn=block_fn,
            block_sizes=_get_block_sizes(resnet_size), block_strides=[1, 2, 2, 2], final_size=final_size,
            data_format=data_format)


def _get_block_sizes(resnet_size):
    """The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.
  """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        msg = 'Could not find layers for selected Resnet size.\nSize received: {}; sizes allowed: {}.'
        err = (msg.format(resnet_size, choices.keys()))
        raise ValueError(err)
