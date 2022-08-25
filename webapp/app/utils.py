import io
import os
import json
import base64
import requests
import numpy as np
from PIL import Image
from urllib import request
from app.models import NI4OSResult, NI4OSData

WIDTH = 256
HEIGHT = 256

def parse_response(json_response, task='classification'):
    json_response = json_response.json()
    json_response = json_response['predictions']

    response = []

    for i, prediction in enumerate(json_response):
        #preds = prediction['probabilities']

        keys = np.array(prediction['classnames'])
        values = np.array(prediction['probabilities'])
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
        elif task.lower().startswith('patches'):
            top_keys = top_keys[top_values>10]
            top_values = top_values[top_values>10]

        response.append(dict(zip(top_keys, top_values)))

    return response

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
        data_bytes = data.read()
        data_enc = base64.b64encode(data_bytes).decode('utf-8')
        req['instances'].append({'b64': data_enc})

        if not task.lower().startswith('patches'):
            result.append(NI4OSResult(data_enc, data.mimetype))

    data_to_send = json.dumps(req)

    if task.lower() == 'classification':
        json_response = requests.post('http://localhost/upload-api',
                                    headers = headers,
                                    data=data_to_send)
    elif task.lower() == 'tagging':
        json_response = requests.post('http://localhost/multilabel-upload-api',
                                    headers = headers,
                                    data=data_to_send)
    elif task.lower() == 'patches classification':
        json_response = requests.post('http://localhost/upload-api-patches',
                                    headers=headers,
                                    data=data_to_send)
    elif task.lower() == 'patches again':
        json_response = requests.post('http://localhost/upload-api-patches',
                                    headers=headers,
                                    data=data_to_send)

    response = parse_response(json_response, task)

    if task.lower().startswith('patches'):
        ROOT = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(ROOT, 'clc_class_colors.json')) as f:
            clc_class_colors = json.load(f)

        for data in forms_data:
            img = Image.open(data)
            newheight = img.height // HEIGHT * HEIGHT
            newwidth = img.width // WIDTH * WIDTH
            img = img.crop((0, 0, newwidth, newheight))
            labelmap = np.zeros((img.height, img.width, 3), dtype='uint8')
            k = 0
            for j in range(0, img.height-HEIGHT+1, HEIGHT):
                for i in range(0, img.width-WIDTH+1, WIDTH):
                    patch = img.crop((i, j, i+WIDTH, j+HEIGHT))
                    patch_jpeg = io.BytesIO()
                    patch.save(patch_jpeg, 'JPEG')
                    result.append(NI4OSResult(base64.b64encode(patch_jpeg.getvalue()).decode('utf-8'), data.mimetype))
                    
                    if task.lower() == 'patches classification':
                        cls = list(response[k].keys())[0]
                        color_code = clc_class_colors[cls]
                        labelmap[j:j+HEIGHT, i:i+WIDTH, :] = color_code
                        k += 1

            if task.lower() == 'patches classification':
                labeled = Image.fromarray(labelmap)
                labeled = Image.blend(labeled, img, 0.5)
                labeled_jpeg = io.BytesIO()
                labeled.save(labeled_jpeg, 'JPEG')
                labeled_jpeg.seek(0)
                labeled_jpeg = base64.b64encode(labeled_jpeg.getvalue()).decode('utf-8')    
             
    for k, res in enumerate(response):
        result[k].results = res

    return result, labeled_jpeg
