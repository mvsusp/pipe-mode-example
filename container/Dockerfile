ARG VERSION
FROM tensorflow/tensorflow:$VERSION

RUN pip install boto3

ADD . /

ENTRYPOINT ["python", "/imagenet_main.py"]
