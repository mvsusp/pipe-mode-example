#!/usr/bin/env python
import sagemaker
from os.path import join
from sagemaker.tensorflow import TensorFlow

sagemaker_session = sagemaker.Session()

if __name__ == '__main__':
    # Setting the input mode to Pipe streams the input channels to the container instead of downloading the entire file.
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container
    input_mode = 'Pipe'

    hyperparameters = {'batch_size': 42, 'resnet_size': 34, 'multi_gpu': False}
    estimator = TensorFlow(input_mode=input_mode, entry_point='imagenet_main.py', source_dir='source_dir',
                           role='SageMakerRole', training_steps=1000000000000000000, evaluation_steps=100,
                           hyperparameters=hyperparameters, train_instance_count=1,
                           train_instance_type='ml.p3.2xlarge', sagemaker_session=sagemaker_session)

    s3_bucket = 's3://{}'.format(sagemaker_session.default_bucket())

    # Let's create one input channel for training, and one input channel for evaluation.
    # Both channels have 1000 TF record files with 20 mb each. Each TF record files has 42 TF serialized Examples.
    # The TF Example is expected to contain the features:
    #       - image/encoded (a JPEG-encoded string)
    #       - image/class/label (int)

    # When the training job starts, SageMaker will create a stream channel for each folder.
    # All files from training will be appended in one single file named training_0, with 42000 images and size of 20 GB.
    # All files from validation will be appended in one single file named validation_0, with 42000 images and size of
    # 20 GB.
    data_dir = 'data/fake-imagenet'
    inputs = {'training': join(s3_bucket, data_dir, 'training'), 'validation': join(s3_bucket, data_dir, 'validation')}

    estimator.fit(inputs, run_tensorboard_locally=True)
