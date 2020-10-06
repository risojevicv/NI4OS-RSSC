#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Build convnet architecture.

Created on Thu Oct 26 14:57:02 2017

@author: vlado
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

def build_net(arch, nr_classes, weights_path=None, trainable=True):
    # Build a convnet model
    
    if arch == 'vgg':      
        from tensorflow.keras.applications.vgg16 import VGG16
        base_net = VGG16(include_top=False, 
                         weights='imagenet', 
                         input_shape=(256, 256, 3))
    elif arch == 'resnet':      
        from tensorflow.keras.applications.resnet50 import ResNet50
        base_net = ResNet50(include_top=False, 
                            weights='imagenet',
                            input_shape=(256, 256, 3))
    elif arch =='inception_v3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        base_net = InceptionV3(include_top=False,
                               weights='imagenet',
                               input_shape=(256, 256, 3))
    else:
        raise NotImplementedError('Valid architectures are: vgg, resnet, inception_v3!')
        
    if weights_path is not None:
        base_net.load_weights(weights_path)

    if not trainable:
        for l in base_net.layers:
            l.trainable = False

    inp = base_net.input
    fea = base_net.output
    fea = GlobalAveragePooling2D()(fea)
    pred = Dense(nr_classes, activation='softmax')(fea)
    classifier = Model(inp, pred)
    
    return classifier
