import os
import json
import tensorflow as tf

class ClassifyJPEG(tf.Module):
    def __init__(self):
        super(ClassifyJPEG, self).__init__()
        tf.keras.backend.set_learning_phase(0)

        class_index = json.load(open('mlrsnet_labels_index.json'))
        self.class_index = tf.constant(list(class_index.values()),
                                       dtype=tf.string,
                                       shape=(1, 60))

        self.clf = tf.keras.models.load_model('models/rssc-resnet50-MLRSNet80_multilabel-chocolate-wave-6-2021-01-07.h5')

    @tf.function(input_signature=[tf.TensorSpec([None,], tf.string)])
    def __call__(self, string_inp):
        imgs_map = tf.map_fn(
                             tf.io.decode_image,
                             string_inp,
                             dtype=tf.uint8)
        imgs_map.set_shape((None, None, None, 3))
        imgs = tf.image.resize(imgs_map, [256, 256])
        img_float = tf.cast(imgs, dtype=tf.float32) / 255.0

        return {'preds': self.clf(img_float),
                'classnames': tf.tile(self.class_index, [len(imgs), 1])}

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

