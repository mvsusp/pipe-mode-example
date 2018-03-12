from __future__ import absolute_import

import logging


format = '%(asctime)s %(levelname)s - %(name)s - %(message)s'

logging.basicConfig(format=format)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)


def create_logger():
    return logging.getLogger('sagemaker-containers')
