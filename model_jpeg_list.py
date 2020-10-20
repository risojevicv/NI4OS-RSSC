emport os
import json
import tensorflow as tf

class ClassifyJPEG(tf.Module):
    def __init__(self):
        super(ClassifyJPEG, self).__init__()
        tf.keras.backend.set_learning_phase(0)

        class_index = json.load(open('nwpu_class_index.json'))
        self.class_index = tf.constant(list(class_index.values()),
                                       dtype=tf.string,
                                       shape=(1, 45))

        base_model = tf.keras.applications.resnet50.ResNet50(weights=None,
                                                     include_top=False,
                                                     input_shape=(256, 256, 3))
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        out = tf.keras.layers.Dense(45, activation='softmax')(x)

        self.clf = tf.keras.models.Model(base_model.input, out)
        self.clf.load_weights('resnet50_nwpu_adam_random_crop_batch100_epochs144.h5')

    @tf.function(input_signature=[tf.TensorSpec([None,], tf.string)])
    def __call__(self, string_inp):
        imgs_map = tf.map_fn(
                             tf.io.decode_image,
                             string_inp,
                             dtype=tf.uint8)
        imgs_map.set_shape((None, None, None, 3))
        imgs = tf.image.resize(imgs_map, [256, 256])
        img_float = tf.cast(imgs, dtype=tf.float32) / 255.0

        return {'scores': self.clf(img_float),
                'classnames': tf.tile(self.class_index, [len(imgs), 1])}

version = 1
export_path = os.path.join('models', str(version))

#tf.config.threading.set_intra_op_parallelism_threads(0)
#tf.config.threading.set_inter_op_parallelism_threads(0)

module_output = ClassifyJPEG()
call_output = module_output.__call__.get_concrete_function(tf.TensorSpec([None,], tf.string))
tf.saved_model.save(
        module_output,
        export_path,
        signatures={'serving_default': call_output})

