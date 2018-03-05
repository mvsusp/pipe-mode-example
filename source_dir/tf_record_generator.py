import logging
import os
import struct
import time

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TFRecordDatasetGenerator(object):

    def __init__(self, channel_name, buffer_size):
        """ Generator that reads TF records from the stream.

        Example:
            generator = TFRecordDatasetGenerator('training_0', 21)
            dataset = Dataset.from_generator(generator, tf.string)
            iterator = dataset.make_one_shot_iterator()

            sess = tf.Session()
            tf_serialized_example = iterator.get_next()

            feature_map = {'image/filename': tf.FixedLenFeature([], dtype=tf.string),
                           'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64)}

            example_tensor = tf.parse_single_example(tf_serialized_example, feature_map)

            tf_example_map = sess.run(example_tensor)

            image_filename, label = tf_example_map['image/filename'], tf_example_map['image/class/label']

            print('TF examples contains image {} and label {}'.format(image_filename, label))

        Args:
            channel_name: (string) containing the channel name
            buffer_size: (int) number of records that will be cached. Higher buffer size means that more records will
            be written in memory from the stream reducing time spent waiting on I/O. Lower buffer size means less
            memory consumption.

        """
        self._stream_dir = '/opt/ml/input/data/'
        self._epoch = 0
        self._channel_name = channel_name
        self._buffer_size = buffer_size
        self._queue = []

    def __call__(self, *args, **kwargs):
        """ Iterator of TF serialized examples.

        Every time next() is called, it ensures that {buffer_size} records are cached in the queue.

        As data is read from the stream, more data is streamed in from S3.
        For better performance, read the maximum possible in advance without crossing memory constraints.
        """
        self._wait_until_stream_exists()

        with open(self._stream_file_path, 'r') as stream:
            while self._refill_queue(stream):
                yield self._queue.pop()

    def _read_example(self, stream):
        """Read next tf serialized example from the stream and returns it as a string.

        Format of a single record:
            uint64    length
            uint32    masked crc of length
            byte      data[length]
            uint32    masked crc of data
            https://github.com/tensorflow/tensorflow/blob/49c20c5814dd80f81ced493d362d374be9ab0b3e/tensorflow/core/lib/io/record_writer.cc#L101

        Returns:
            None if stream is empty.
            string containing tf serialized example read from the stream.
        """

        length = self._read_c_int_64(stream)

        if not length:
            return None

        crc_length = self._read_c_int_32(stream)
        example = stream.read(length)
        crc_data = self._read_c_int_32(stream)
        return example

    def _refill_queue(self, fifo):
        """Ensures that queue contains {buffer_size} items

        Returns:
            length of the Queue
        """
        for i in xrange(len(self._queue), self._buffer_size):
            example = self._read_example(fifo)

            if example:
                self._queue.append(example)

        msg = 'Generator for channel {} has queue size of {} records'
        logger.debug(msg.format(self._channel_name, len(self._queue)))
        return len(self._queue)

    @property
    def _stream_file_path(self):
        """

        Returns: path to the current stream.
        """
        return os.path.join(self._stream_dir, '{}_{}'.format(self._channel_name, self._epoch))

    def _wait_until_stream_exists(self):
        """ Wait for the stream to be created
        """
        logger.info("Wait until stream is available: {0}".format(self._stream_file_path))
        while not os.path.exists(self._stream_file_path):
            time.sleep(.1)

    @staticmethod
    def _read_c_struct(stream, format):
        """ Read a C struct from the stream.
        """
        struct_size = struct.calcsize(format)
        buffer = stream.read(struct_size)

        if not buffer:
            return None

        return struct.unpack(format, buffer)[0]

    def _read_c_int_64(self, stream):
        """ Read a C int 64 from the stream
        """
        return self._read_c_struct(stream, '<Q')

    def _read_c_int_32(self, stream):
        """ Read a C int 32 from the stream
        """
        return self._read_c_struct(stream, '<I')
