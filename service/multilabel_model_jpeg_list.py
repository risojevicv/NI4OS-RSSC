import os
import json
import numpy as np
import tensorflow as tf

class ClassifyJPEG(tf.Module):
    def __init__(self):
        super(ClassifyJPEG, self).__init__()
#        tf.keras.backend.set_learning_phase(0)

        class_index = json.load(open('mlrsnet_labels_index.json'))
        self.class_index = tf.constant(list(class_index.values()),
                                       dtype=tf.string,
                                       shape=(1, 60))

        self.clf = tf.keras.models.load_model('models/rssc_resnet50_imagenet-MLRSNet80_multilabel-ft.h5')

    def __load_preprocess(self, inp):
        img = tf.io.decode_image(inp)
        img.set_shape((None, None, 3))
        img = tf.image.resize(img, [256, 256])
        means = tf.constant(np.reshape([123.68, 116.779, 103.939], (1, 1, 3)),
                            dtype=tf.float32)
        img = tf.math.subtract(img, means)       

        return img
        
    @tf.function(input_signature=[tf.TensorSpec([None,], tf.string)])
    def __call__(self, string_inp):
        imgs = tf.map_fn(
                         self.__load_preprocess,
                         string_inp,
                         fn_output_signature=tf.float32)

        return {'probabilities': self.clf(imgs),
                'classnames': tf.tile(self.class_index, [tf.shape(imgs)[0], 1])}

version = 1
export_path = os.path.join('models/multilabel_rssc', str(version))

#tf.config.threading.set_intra_op_parallelism_threads(0)
#tf.config.threading.set_inter_op_parallelism_threads(0)

module_output = ClassifyJPEG()
call_output = module_output.__call__.get_concrete_function(tf.TensorSpec([None,], tf.string))
tf.saved_model.save(
        module_output,
        export_path,
        signatures={'serving_default': call_output})

