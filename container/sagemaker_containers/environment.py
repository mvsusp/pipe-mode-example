from __future__ import absolute_import

import importlib
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import tarfile

import boto3
from six.moves.urllib.parse import urlparse
from .logging import create_logger

logger = create_logger()

BASE_DIRECTORY = "/opt/ml"
USER_SCRIPT_NAME_PARAM = "sagemaker_program"
USER_SCRIPT_ARCHIVE_PARAM = "sagemaker_submit_directory"
CLOUDWATCH_METRICS_PARAM = "sagemaker_enable_cloudwatch_metrics"
CONTAINER_LOG_LEVEL_PARAM = "sagemaker_container_log_level"
JOB_NAME_PARAM = "sagemaker_job_name"
CURRENT_HOST_ENV = "CURRENT_HOST"
JOB_NAME_ENV = "JOB_NAME"
USE_NGINX_ENV = "SAGEMAKER_USE_NGINX"
SAGEMAKER_REGION_PARAM_NAME = 'sagemaker_region'
SAGEMAKER_EXECUTION_MODE = 'sagemaker_execution_mode'
SCRIPT_MODE = 'script'
FUNCTION_MODE = 'function'
FRAMEWORK_MODULE_NAME = "CONTAINER_MODULE_NAME"


class ContainerEnvironment(object):
    """Provides access to common aspects of the container environment, including
    important system characteristics, filesystem locations, and configuration settings.
    """

    def __init__(self, base_dir=BASE_DIRECTORY):
        self._base_dir = base_dir
        "The current root directory for SageMaker interactions (``/opt/ml`` when running in SageMaker)."

        self.model_dir = os.path.join(base_dir, "model")
        "The directory to write model artifacts to so they can be handed off to SageMaker."

        self.code_dir = os.path.join(base_dir, "code")
        "The directory where user-supplied code will be staged."

        self.available_cpus = self._get_available_cpus()
        "The number of cpus available in the current container."

        self.available_gpus = self._get_available_gpus()
        "The number of gpus available in the current container."

        # subclasses will override
        self._user_script_name = None
        "The filename of the python script that contains user-supplied training/hosting code."

        # subclasses will override
        self._user_script_archive = None
        "The S3 location of the python code archive that contains user-supplied training/hosting code"

        self._enable_cloudwatch_metrics = False
        "Report system metrics to CloudWatch? (default = False)"

        # subclasses will override
        self._container_log_level = None
        "The logging level for the root logger."

        # subclasses will override
        self._sagemaker_region = None
        "The current AWS region."

    def download_user_module(self):
        """Download user-supplied python archive from S3.
        """
        tmp = os.path.join(tempfile.gettempdir(), "script.tar.gz")
        download_s3_resource(self._user_script_archive, tmp)
        untar_directory(tmp, self.code_dir)

    def import_user_module(self):
        """Import user-supplied python module.
        """
        sys.path.insert(0, self.code_dir)

        script = self._user_script_name
        if script.endswith(".py"):
            script = script[:-3]

        user_module = importlib.import_module(script)
        return user_module

    @staticmethod
    def _get_available_cpus():
        return multiprocessing.cpu_count()

    @staticmethod
    def _get_available_gpus():
        gpus = 0
        try:
            output = subprocess.check_output(["nvidia-smi", "--list-gpus"]).decode('utf-8')
            gpus = sum([1 for x in output.split('\n') if x.startswith('GPU ')])
        except Exception as e:
            logger.debug("exception listing gpus (normal if no nvidia gpus installed): %s" % str(e))

        return gpus

    @staticmethod
    def _load_config(path):
        with open(path, 'r') as f:
            return json.load(f)


def parse_s3_url(url):
    """ Returns an (s3 bucket, key name/prefix) tuple from a url with an s3 scheme
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme != "s3":
        raise ValueError("Expecting 's3' scheme, got: {} in {}".format(parsed_url.scheme, url))
    return parsed_url.netloc, parsed_url.path.lstrip('/')


def download_s3_resource(source, target):
    """ Downloads the s3 object source and stores in a new file with path target.
    """
    print("Downloading {} to {}".format(source, target))
    s3 = boto3.resource('s3')

    script_bucket_name, script_key_name = parse_s3_url(source)
    script_bucket = s3.Bucket(script_bucket_name)
    script_bucket.download_file(script_key_name, target)

    return target


def untar_directory(tar_file_path, extract_dir_path):
    with open(tar_file_path, 'rb') as f:
        with tarfile.open(mode='r:gz', fileobj=f) as t:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(t, path=extract_dir_path)
