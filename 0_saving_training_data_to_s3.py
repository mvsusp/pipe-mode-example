#!/usr/bin/env python
import argparse

import boto3
import os
import sagemaker

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

session = boto3.Session()
s3_client = session.resource('s3')
sagemaker_session = sagemaker.Session(session)


def upload_file(channel, index, bucket, key, local_path):
    previous = (index - 1) * 42
    current = (index * 42) - 1

    file_name = '{}-{}-of-{}'.format(channel, str(previous).zfill(5), str(current).zfill(5))
    s3_key = os.path.join(key, channel, file_name)
    s3_client.Object(bucket, s3_key).upload_file(local_path)
    print('Uploaded file {} to {}/{}'.format(os.path.basename(local_path), bucket_name, s3_key))


if __name__ == '__main__':
    msg = """This script copies the tf records files located under the data folder N times to default S3 bucket. 
    """
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('--S3-key', help='s3 location where the files will be written.', default='data/fake-imagenet',
                        type=str)
    parser.add_argument('-N', help='Number of copies', default=1000)
    args = parser.parse_args()

    bucket_name = sagemaker_session.default_bucket()

    for i in xrange(1, args.N):
        upload_file('training', i, bucket_name, args.S3_key, os.path.join(data_dir, 'tf-training-records'))
        upload_file('validation', i, bucket_name, args.S3_key, os.path.join(data_dir, 'tf-validation-records'))
