import requests
from app.models import NI4OSResult, NI4OSData
import numpy as np
import base64
import json


def parse_response(json_response, task='classification'):
    json_response = json_response.json()
    json_response = json_response['predictions']

    response = []

    for i, prediction in enumerate(json_response):
        preds = prediction['preds']
        #idxs = np.array(preds).argsort()[:-6:-1]

        keys = np.array(prediction['classnames'])
        values = np.array(prediction['preds'])
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

    for data in forms_data:
        data_bytes = base64.b64encode(data.read()).decode('utf-8')
        result.append(NI4OSResult(data_bytes, data.mimetype))
        req['instances'].append({'b64': data_bytes})

    data_to_send = json.dumps(req)

    if task.lower() == 'classification':
        json_response = requests.post('http://localhost/upload-api',
                                    headers = headers,
                                    data=data_to_send)
    elif task.lower() == 'tagging':
        json_response = requests.post('http://localhost/multilabel-upload-api',
                                    headers = headers,
                                    data=data_to_send)

    response = parse_response(json_response, task)

    for i, out in enumerate(response):
        result[i].results = out

    return result
