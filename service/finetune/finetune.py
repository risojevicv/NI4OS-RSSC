#!/usr/bin/env python
# coding: utf-8

"""

Convnet training on NWPU.

Created on Thu Jul  4 23:37:36 2019

@author: vlado
"""
import os
os.environ['PYTHONHASHSEED'] = '42'
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

import glob
from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications.resnet_v2 import ResNet50V2
#from resnet50_wd import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import datetime
import utils
from build_net import build_net

class LRScheduleSteps(Callback):
  def __init__(self, opt, lr, w, p, factor=10.):
    super(LRScheduleSteps, self).__init__()
    self.optimizer = opt
    self.lr = lr
    self.w = w
    self.p = p
    self.factor = factor

    self.step = 0

  def on_batch_end(self, batch, logs=None):
    logs = logs or {}
    
    if self.step < self.w:
      lr = self.lr / self.w * self.step
    else:
      lr = K.get_value(self.optimizer.lr)
      for step in self.p:
        if step == self.step:
          lr = lr / self.factor

    K.set_value(self.optimizer.lr, lr)    
    self.step += 1


def build_image_list(img_path, max_samples_per_class=-1):
    
    le = LabelEncoder()
    classes = os.scandir(img_path)
    all_imgs = []
    all_labels = []
    for cl in classes:
        fullpath = os.path.join(img_path, cl)
        if cl.is_dir() and not cl.name.startswith('.'):
            files = glob.glob(fullpath + os.path.sep + '*.jpg')
            if max_samples_per_class > 0:
                num_samples = min(len(files), max_samples_per_class)
            else:
                num_samples = len(files)
            for k in range(num_samples):
                all_imgs.append(os.path.join(fullpath, files[k]))
                all_labels.append(cl.name)
                
    all_labels_id = le.fit_transform(all_labels)
        
    return all_imgs, all_labels_id, le

def split_data(all_imgs, all_labels_id, val_ratio=0.2):

    x_train, x_val, y_train, y_val = train_test_split(all_imgs, all_labels_id,
                                                       test_size=val_ratio,
                                                       stratify=all_labels_id)
    data_partition = {'train': {'filename': x_train, 'label': y_train},
                      'val': {'filename': x_val, 'label': y_val}}
       
    return data_partition

class LoadPreprocessImage():
  def __init__(self, load_size=(256, 256), dim=(224, 224),
               crop_size=(32, 256)):
    self.load_size = load_size
    self.dim = dim
    self.crop_size = crop_size

  def __call__(self, record):
    image = tf.io.read_file(record['filename'])
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, self.load_size)
    image = tf.image.random_crop(image, self.dim+(3,))
    n = tf.random.uniform((), maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=n)
    if tf.random.uniform(()) > 0.5:
      image = tf.image.flip_left_right(image)
    if tf.random.uniform(()) > 0.5:
      image = tf.image.flip_up_down(image)
    
    image = tf.cast(image, tf.float32)
    means = tf.constant(np.reshape([123.68, 116.779, 103.939], (1, 1, 3)), 
                        dtype=tf.float32)
    image = tf.math.subtract(image, means)
    
    return image, record['label']


class LoadPreprocessImageVal():
  def __init__(self, load_size=(256, 256), dim=(224, 224)):
    self.load_size = load_size
    self.dim = dim

  def __call__(self, record):
    image = tf.io.read_file(record['filename'])
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, self.load_size)
    image = tf.image.central_crop(image, self.dim[0]/self.load_size[0])
    
    image = tf.cast(image, tf.float32)
    means = tf.constant(np.reshape([123.68, 116.779, 103.939], (1, 1, 3)), 
                        dtype=tf.float32)
    image = tf.math.subtract(image, means)    
    
    return image, record['label']


args = utils.get_parser().parse_args()

all_imgs, all_labels_id, le = build_image_list(args.data, max_samples_per_class=10)
nr_classes = len(le.classes_)

data_partition = split_data(all_imgs, all_labels_id, val_ratio=0.2)
nr_training_imgs = len(data_partition['train']['filename'])
    
train_list_ds = tf.data.Dataset.from_tensor_slices(data_partition['train'])
train_ds = train_list_ds.shuffle(nr_training_imgs) \
                        .map(LoadPreprocessImage(load_size=(292, 292),
                                                 dim=(256, 256),
                                                 crop_size=(32, 256)),
                                                 num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                        .batch(args.batch_size) \
                        .prefetch(tf.data.experimental.AUTOTUNE)
val_list_ds = tf.data.Dataset.from_tensor_slices(data_partition['val'])
val_ds = val_list_ds.map(LoadPreprocessImageVal(load_size=(292, 292),
                                                dim=(256, 256)),
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                    .batch(args.batch_size) \
                    .prefetch(tf.data.experimental.AUTOTUNE)
    
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    clf = build_net(args.arch, nr_classes)
    
    opt = optimizers.Adam(lr=1e-5)
    clf.compile(opt, 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        
lrsched = LRScheduleSteps(opt, 1e-4, 1260, [12600, 17640, 22680], factor=5.)
checkpoint = ModelCheckpoint('work/checkpoint.h5',
                             save_freq=10*args.batch_size,
                             save_best_only=False)
    
h = clf.fit(train_ds,
            epochs=3,
            callbacks=[checkpoint, lrsched],
            validation_data=val_ds,
            verbose=2)
                      
dataset_name = os.path.basename(os.path.normpath(args.data))
clf.save_weights('weights/rssc_{}_imagenet_{}_{}.h5'.format(
                                                      args.arch,
                                                      dataset_name,
                                                      datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')))
    
loss, acc = clf.evaluate(val_ds)
    
print('Validation accuracy: {:.2f}'.format(100*acc))

