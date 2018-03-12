from __future__ import absolute_import

import json
import os

from sagemaker_containers.environment import ContainerEnvironment, BASE_DIRECTORY, USER_SCRIPT_NAME_PARAM, \
    USER_SCRIPT_ARCHIVE_PARAM, CLOUDWATCH_METRICS_PARAM, CONTAINER_LOG_LEVEL_PARAM, JOB_NAME_PARAM, JOB_NAME_ENV, \
    CURRENT_HOST_ENV
from .logging import create_logger

logger = create_logger()

HYPERPARAMETERS_FILE = "hyperparameters.json"
RESOURCE_CONFIG_FILE = "resourceconfig.json"
INPUT_DATA_CONFIG_FILE = "inputdataconfig.json"
S3_URI_PARAM = 'sagemaker_s3_uri'


class TrainingEnvironment(ContainerEnvironment):
    """Provides access to aspects of the container environment relevant to training jobs.
    """

    def __init__(self, base_dir=BASE_DIRECTORY):
        super(TrainingEnvironment, self).__init__(base_dir)
        self.input_dir = os.path.join(self._base_dir, "input")
        "The base directory for training data and configuration files."

        self.input_config_dir = os.path.join(self.input_dir, "config")
        "The directory where standard SageMaker configuration files are located."

        self.output_dir = os.path.join(self._base_dir, "output")
        "The directory where training success/failure indications will be written."

        self.resource_config = self._load_config(os.path.join(self.input_config_dir, RESOURCE_CONFIG_FILE))
        "dict of resource configuration settings."

        self.hyperparameters = self._load_hyperparameters(os.path.join(self.input_config_dir, HYPERPARAMETERS_FILE))
        "dict of hyperparameters that were passed to the CreateTrainingJob API."

        self.current_host = self.resource_config.get('current_host', '')
        "The hostname of the current container."

        self.hosts = self.resource_config.get('hosts', [])
        "The list of hostnames available to the current training job."

        host_dir = self.current_host if len(self.hosts) > 1 else ''
        self.output_data_dir = os.path.join(self.output_dir, "data", host_dir)
        "The dir to write non-model training artifacts (e.g. evaluation results) which will be retained by SageMaker. "

        self.channels = self._load_config(os.path.join(self.input_config_dir, INPUT_DATA_CONFIG_FILE))
        "dict of training input data channel name to directory with the input files for that channel."

        self.channel_dirs = {channel: self._get_channel_dir(channel) for channel in self.channels}

        self._user_script_name = self.hyperparameters.get(USER_SCRIPT_NAME_PARAM, '')
        self._user_script_archive = self.hyperparameters.get(USER_SCRIPT_ARCHIVE_PARAM, '')

        self._enable_cloudwatch_metrics = self.hyperparameters.get(CLOUDWATCH_METRICS_PARAM, False)
        self.container_log_level = self.hyperparameters.get(CONTAINER_LOG_LEVEL_PARAM)

        os.environ[JOB_NAME_ENV] = self.hyperparameters.get(JOB_NAME_PARAM, '')
        os.environ[CURRENT_HOST_ENV] = self.current_host

        self.distributed = len(self.hosts) > 1

    def _load_hyperparameters(self, path):
        serialized = self._load_config(path)
        return self._deserialize_hyperparameters(serialized)

    @staticmethod
    def _deserialize_hyperparameters(hp):
        return {k: json.loads(v) for (k, v) in hp.items()}

    def write_success_file(self):
        self.write_output_file('success')

    def write_failure_file(self, message):
        self.write_output_file('failure', message)

    def write_output_file(self, file_name, message=None, base_dir=None):
        base_dir = base_dir or BASE_DIRECTORY
        output_dir = os.path.join(base_dir, "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        message = message if message else 'Training finished with {}'.format(file_name)

        print(message)
        with open(os.path.join(output_dir, file_name), 'a') as fd:
            fd.write(message)

    def _get_channel_dir(self, channel):
        """ Returns the directory containing the channel data file(s).

        This is either:

        - <self.base_dir>/input/data/<channel> OR
        - <self.base_dir>/input/data/<channel>/<channel_s3_suffix>

        Where channel_s3_suffix is the hyperparameter value with key <S3_URI_PARAM>_<channel>.

        The first option is returned if <self.base_dir>/input/data/<channel>/<channel_s3_suffix>
        does not exist in the file-system or <S3_URI_PARAM>_<channel> does not exist in
        self.hyperparmeters. Otherwise, the second option is returned.

        TODO: Refactor once EASE downloads directly into /opt/ml/input/data/<channel>
        TODO: Adapt for Pipe Mode

        Returns:
            (str) The input data directory for the specified channel.
        """
        channel_s3_uri_param = "{}_{}".format(S3_URI_PARAM, channel)
        if channel_s3_uri_param in self.hyperparameters:
            channel_s3_suffix = self.hyperparameters.get(channel_s3_uri_param)
            channel_dir = os.path.join(self.input_dir, 'data', channel, channel_s3_suffix)
            if os.path.exists(channel_dir):
                return channel_dir
        return os.path.join(self.input_dir, 'data', channel)
