import tensorflow as tf
import requests
import base64
import numpy as np

from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2


#IMAGE_PATH = '/home/vlado/dl/data/NWPU-RESISC45/airplane/airplane_001.jpg'
IMAGE_PATH = 'airplane_001.jpg'
NUM_RECORDS = 100

def main():
    """Generate TFRecords for warming up."""

    with tf.io.TFRecordWriter("tf_serving_warmup_requests") as writer:
        with open(IMAGE_PATH, 'rb') as f:
#            image_data = base64.b64encode(f.read()).decode('utf-8')
            image_data = f.read()
        predict_request = predict_pb2.PredictRequest()
        predict_request.model_spec.name = 'resnet'
        predict_request.model_spec.signature_name = 'serving_default'
        predict_request.inputs['string_inp'].CopyFrom(
            tensor_util.make_tensor_proto(image_data,
                                          shape=[1,]))        
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=predict_request))
        for r in range(NUM_RECORDS):
            writer.write(log.SerializeToString())    

if __name__ == "__main__":
    main()
