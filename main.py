#!/usr/bin/env python
from argparse import ArgumentParser
from os import path
from subprocess import check_call, check_output

import sagemaker
from sagemaker.estimator import Estimator

sagemaker_session = sagemaker.Session()


class ArgParser(ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--gpu', action='store_true', help='If set, uses GPU image')
        self.add_argument('--normal-mode', action='store_true', help='If set, uses SageMaker training')
        self.add_argument('--full-data', action='store_true', help='If set, uses entire dataset')
        self.add_argument('--local-data', action='store_true', help='If set, uses local data')
        self.add_argument('--push', action='store_true', help='If set, pushed image to args.ecr-repository')
        self.add_argument('--ecr-repository', help='ECR repo where images will be pushed', default='add-ecr-repo-here')
        self.add_argument('--input-mode', default='Pipe')


def build_image(name, version):
    cmd = 'docker build -t {} --build-arg VERSION={} -f Dockerfile .'.format(name, version).split(' ')
    check_call(cmd, cwd='container')


def push_image(name):
    cmd = 'aws ecr get-login --no-include-email --region us-west-2'.split(' ')
    login = check_output(cmd).strip()

    check_call(login.split(' '))

    check_call(cmd)
    cmd = 'docker push {}'.format(name).split(' ')
    check_call(cmd)


if __name__ == '__main__':
    args = ArgParser().parse_args()

    version = 'latest-gpu' if args.gpu else 'latest'
    train_instance_type = 'ml.p3.2xlarge' if args.gpu else 'ml.c5.9xlarge'

    image_name = '{}:pipe-vs-file-mode'.format(args.ecr_repository)

    build_image(image_name, version)

    if args.push:
        push_image(image_name)

    hyperparameters = {'mode': args.input_mode,
                       'num_parallel_calls': 5, 'resnet_size': 50, 'train_epochs': 10, 'epochs_per_eval': 1,
                       'batch_size': 32, 'multi_gpu': False, 'inter_op_parallelism_threads': 0,
                       'intra_op_parallelism_threads': 0}

    train_volume_size = 30 if args.input_mode == 'Pipe' else 200

    estimator = Estimator(image_name, role='SageMakerRole', train_instance_count=1, train_volume_size=train_volume_size,
                          train_instance_type=train_instance_type, input_mode=args.input_mode,
                          local_mode=not args.normal_mode, hyperparameters=hyperparameters)

    s3_bucket = 's3://{}'.format(sagemaker_session.default_bucket())

    data_dir = 'data/fake-imagenet' if args.full_data else 'data/less_data'

    inputs = {'training': path.join(s3_bucket, data_dir, 'training'),
              'validation': path.join(s3_bucket, data_dir, 'validation')}

    if args.local_data:
        current_dir = path.dirname(path.abspath(__file__))
        inputs = {'training': path.join(current_dir, 'data', 'training'),
                  'validation': path.join(current_dir, 'data', 'validation')}

    estimator.fit(inputs)
