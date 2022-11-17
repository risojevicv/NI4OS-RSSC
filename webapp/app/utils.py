"""
Contains parts of the code adapted from https://github.com/bnsreenu/python_for_microscopists, 
which in turn adapted the original code from the following source. It comes with MIT License 
so please mention the original reference when sharing.

The original code has been modified to fix a couple of bugs and chunks of code
unnecessary for smooth tiling are removed. 

# MIT License
# Copyright (c) 2017 Vooban Inc.
# Coded by: Guillaume Chevalier
# Source to original code and license:
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/LICENSE

"""

import io
import os
import json
import base64
import requests
import numpy as np
from PIL import Image
from app import ann_index, ann_index_multilabel
from urllib import request
from app.models import NI4OSResult, NI4OSData

WIDTH = 256
HEIGHT = 256
ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT, 'clc_class_colors.json')) as f:
    CLC_CLASS_COLORS = json.load(f)
NUM_CLC_CLASSES = len(CLC_CLASS_COLORS)

def triang(M, sym=True):

    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(1, (M + 1) // 2 + 1)
    if M % 2 == 0:
        w = (2 * n - 1.0) / M
        w = np.r_[w, w[::-1]]
    else:
        w = 2 * n / (M + 1.0)
        w = np.r_[w, w[-2::-1]]

    if not sym and not odd:
        w = w[:-1]
    return w

def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)      #SREENI: Changed from 3, 3, to 1, 1 
        wind = wind * wind.transpose(1, 0, 2)
        cached_2d_windows[key] = wind
    return wind

def recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)
    subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])

    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len-window_size+1, step):
        for j in range(0, pady_len-window_size+1, step):                #SREENI: Changed padx to pady (Bug in original code)
            windowed_patch = subdivs[a]
            y[i:i+window_size, j:j+window_size] = y[i:i+window_size, j:j+window_size] + windowed_patch
            a += 1
    return y / (subdivisions ** 2)

def is_ood(features, task='classification'):
    # thresholds estimated using AID as ID and Food-5k as OOD
    # if task.lower() == 'classification':
    if 'classification' in task.lower():
        thr = 0.787 # TNR=0.9, FPR=0.1486
        ann = ann_index
    elif task.lower() == 'tagging':
        thr = 0.89 # TNR=0.83, FPR=0.157
        ann = ann_index_multilabel
    
    ood = []
    for v in features:
        _, d = ann.get_nns_by_vector(v, 3, search_k=-1, include_distances=True)
        ood.append(np.mean(d) > thr)
    
    return ood

def parse_response(json_response, task='classification'):

    with open(os.path.join(ROOT, 'nwpu_clc_map.json')) as f:
        class_mapping = json.load(f)
    class_mapping = list(class_mapping.values())

    json_response = json_response.json()
    json_response = json_response['predictions']

    response = []
    features = []

    for i, prediction in enumerate(json_response):

        features.append(prediction['features'])
        keys = np.array(prediction['classnames'])
        values = np.array(prediction['probabilities'])
        if task.lower().startswith('patch'):
            values_clc = np.zeros(NUM_CLC_CLASSES)    
            np.add.at(values_clc, class_mapping, values)
            values = values_clc
            keys = np.array(list(CLC_CLASS_COLORS.keys()))
        values *= 100
        values = values.astype(np.uint8)

        idxs = values.argsort()[::-1]
        top_keys = keys[idxs]
        top_values = values[idxs]

        if task.lower() == 'classification':
            top_keys = top_keys[top_values>0]
            top_values = top_values[top_values>0]
        elif task.lower() == 'tagging':
            top_keys = top_keys[top_values>50]
            top_values = top_values[top_values>50]
        elif task.lower() == 'patch-based classification':
            top_keys = top_keys[top_values>10]
            top_values = top_values[top_values>10]
        elif task.lower() == 'patch-based classification (smoothed)':
            top_keys = keys
            top_values = values

        response.append(dict(zip(top_keys, top_values)))

    ood = is_ood(features, task)
    
    return response, ood

def perform_url_request(urls, task='classification'):
    if not isinstance(urls, list):
        urls = [urls]

    headers = {'content-type': 'application/x-www-form-urlencoded'}
    data = 'urls=' + ','.join(urls)

    result = []

    for url in urls:
        result.append(NI4OSResult(url))

    if task.lower() == 'classification':
        json_response = requests.post('http://localhost/url-api',
                                    headers=headers,
                                    data=data)
    elif task.lower() == 'tagging':
        json_response = requests.post('http://localhost/multilabel-url-api',
                                    headers=headers,
                                    data=data)

    response = parse_response(json_response, task)

    for i, out in enumerate(response):
        result[i].results = out

    return result


def perform_upload_request(forms_data, task='classification'):
    headers = {'content-type': 'application/json'}
    req = {'signature_name': 'serving_default', 'instances': []}

    result = []
    labeled_jpeg = None

    for data in forms_data:
        if not task.lower().startswith('patch'):
            data_bytes = data.read()
            data_enc = base64.b64encode(data_bytes).decode('utf-8')
            req['instances'].append({'b64': data_enc})
            result.append(NI4OSResult(data_enc, data.mimetype))
        else:
            img = Image.open(data)
            newheight = img.height // HEIGHT * HEIGHT
            newwidth = img.width // WIDTH * WIDTH
            img = img.crop((0, 0, newwidth, newheight))
            OVERLAP = 2 if 'smoothed' in task.lower() else 1
            for j in range(0, img.height-HEIGHT+1, HEIGHT//OVERLAP):
                for i in range(0, img.width-WIDTH+1, WIDTH//OVERLAP):
                    patch = img.crop((i, j, i+WIDTH, j+HEIGHT))
                    patch_jpeg = io.BytesIO()
                    patch.save(patch_jpeg, 'JPEG')
                    patch_enc = base64.b64encode(patch_jpeg.getvalue()).decode('utf-8')
                    req['instances'].append({'b64': patch_enc})
                    result.append(NI4OSResult(patch_enc, data.mimetype))
            
    data_to_send = json.dumps(req)

    if task.lower() == 'classification':
        json_response = requests.post('http://localhost/upload-api',
                                    headers = headers,
                                    data=data_to_send)
    elif task.lower() == 'tagging':
        json_response = requests.post('http://localhost/multilabel-upload-api',
                                    headers = headers,
                                    data=data_to_send)
    elif task.lower().startswith('patch'):
        json_response = requests.post('http://localhost/upload-api',
                                    headers = headers,
                                    data=data_to_send)

    response, ood = parse_response(json_response, task)

    if task.lower().startswith('patch'):
        labelmap = np.zeros((img.height, img.width, 3), dtype='uint8')
        k = 0
        if 'smoothed' in task.lower():
            probas_patches = []
        for j in range(0, img.height-HEIGHT+1, HEIGHT//OVERLAP):
            for i in range(0, img.width-WIDTH+1, WIDTH//OVERLAP):                 
                if task.lower() == 'patch-based classification':
                    cls = list(response[k].keys())[0]
                    color_code = CLC_CLASS_COLORS[cls]
                    labelmap[j:j+HEIGHT, i:i+WIDTH, :] = color_code
                    k += 1
                elif task.lower() == 'patch-based classification (smoothed)':
                    proba = np.array(list(response[k].values()))
                    probas_patches.append(np.tile(proba, (HEIGHT, WIDTH, 1)))
                    k += 1

        if task.lower().startswith('patch-based classification'):
            if 'smoothed' in task.lower():
                classnames = list(response[0].keys())
                proba_map = recreate_from_subdivs(probas_patches, HEIGHT, OVERLAP, [img.height, img.width, NUM_CLC_CLASSES])
                final_prediction = np.argmax(proba_map, axis=2)
                for cl in range(NUM_CLC_CLASSES):
                    labelmap[final_prediction == cl] = CLC_CLASS_COLORS[classnames[cl]]
            labeled = Image.fromarray(labelmap)
            labeled = Image.blend(labeled, img, 0.5)
            labeled_jpeg = io.BytesIO()
            labeled.save(labeled_jpeg, 'JPEG')
            labeled_jpeg.seek(0)
            labeled_jpeg = base64.b64encode(labeled_jpeg.getvalue()).decode('utf-8')    
             
    for k, (res, ood_flag) in enumerate(zip(response, ood)):
        result[k].results = res
        result[k].ood = ood_flag

    return result, labeled_jpeg, CLC_CLASS_COLORS
