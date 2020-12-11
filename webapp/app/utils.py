import requests
from app.models import NI4OSResult, NI4OSData
import numpy as np
import base64
import json


def parse_response(json_response):
    json_response = json_response.json()
    json_response = json_response['predictions']

    response = []

    for i, prediction in enumerate(json_response):
        preds = prediction['preds']
        idxs = np.array(preds).argsort()[:-6:-1]

        top_keys = np.array(prediction['classnames'])[idxs]
        top_values = np.array(prediction['preds'])[idxs]
        top_values *= 100
        top_values = top_values.astype(np.uint8)
        top_keys = top_keys[top_values>0]
        top_values = top_values[top_values>0]

        response.append(dict(zip(top_keys, top_values)))

    return response

def perform_url_request(urls):
    if not isinstance(urls, list):
        urls = [urls]

    headers = {'content-type': 'application/x-www-form-urlencoded'}
    data = 'urls=' + ','.join(urls)

    result = []

    for url in urls:
        result.append(NI4OSResult(url))

    json_response = requests.post('http://localhost/url-api',
                                  headers=headers,
                                  data=data)

    response = parse_response(json_response)

    for i, out in enumerate(response):
        result[i].results = out

    return result


def perform_upload_request(forms_data):
    headers = {'content-type': 'application/json'}
    req = {'signature_name': 'serving_default', 'instances': []}

    result = []

    for data in forms_data:
        data_bytes = base64.b64encode(data.read()).decode('utf-8')
        result.append(NI4OSResult(data_bytes, data.mimetype))
        req['instances'].append({'b64': data_bytes})

    data_to_send = json.dumps(req)

    json_response = requests.post('http://localhost/upload-api',
                                   headers = headers,
                                   data=data_to_send)

    response = parse_response(json_response)

    for i, out in enumerate(response):
        result[i].results = out

    return result
