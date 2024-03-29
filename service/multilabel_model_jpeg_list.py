import os
import json
import numpy as np
import tensorflow as tf

WIDTH = 256
HEIGHT = 256

class ClassifyJPEG(tf.Module):
    def __init__(self):
        super(ClassifyJPEG, self).__init__()
#        tf.keras.backend.set_learning_phase(0)

        class_index = json.load(open('mlrsnet_labels_index.json'))
        self.class_index = tf.constant(list(class_index.values()),
                                       dtype=tf.string,
                                       shape=(1, 60))

        # self.clf = tf.keras.models.load_model('models/rssc_resnet50_imagenet-MLRSNet80_multilabel-ft.h5')
        base_model = tf.keras.models.load_model('models/rssc_resnet50_MLRSNet80_multilabel.h5')
        backbone = tf.keras.applications.resnet50.ResNet50(weights=None,
                                                    include_top=False,
                                                    input_shape=(256, 256, 3),
                                                    pooling='avg')
        x = backbone.output
        out = tf.keras.layers.Dense(60, activation='sigmoid')(x)

        self.clf = tf.keras.models.Model(backbone.input, [out, x])
        all_weights = base_model.get_weights()
        self.clf.set_weights(all_weights)

    def __load_preprocess(self, inp):
        img = tf.io.decode_image(inp)
        img.set_shape((None, None, 3))
        img = tf.image.resize(img, [HEIGHT, WIDTH])
        img = tf.cast(img, tf.float32) / 255.0
        means = tf.constant(np.reshape([0., 0., 0.], (1, 1, 3)),
                            dtype=tf.float32)
        img = tf.math.subtract(img, means)       

        return img
        
    @tf.function(input_signature=[tf.TensorSpec([None,], tf.string)])
    def __call__(self, string_inp):
        imgs = tf.map_fn(
                         self.__load_preprocess,
                         string_inp,
                         fn_output_signature=tf.float32)

        proba, features = self.clf(imgs)
        return {'probabilities': proba,
                'features': features,
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

