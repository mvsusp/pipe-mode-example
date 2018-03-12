#!/usr/bin/env python

import cv2
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    img_size = 0


    def input_fn(data=None, content_type=None):
        nparr = np.fromstring(data, np.uint8)
        img = cv2.imdecode(nparr, 3)
        #
        img = cv2.resize(img, (img_size, img_size))
        img = img.reshape((1, img_size, img_size, 3))
        #

        estimator = tf.estimator.inputs.numpy_input_fn(
            x={"input_image": img.astype(np.float32)},
            shuffle=True)

        return estimator

    def getRequestData():
        f = open("/Users/mvs/Pictures/estimator.deploy.JPG", "rb")
        img = f.read()
        f.close()
        return img


    payload = getRequestData()
    strin = str(payload)

    print(input_fn(strin)())