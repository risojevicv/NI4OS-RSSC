#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Minimal Flask application

Created on Tue Jan 14 13:12:00 2020

@author: vlado
"""

import requests
import base64
import json
from flask import Flask, escape, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    urls = request.form['urls'].split(',')
    headers = {'content-type': 'application/json'}
    req = {'signature_name': 'serving_default',
           'instances': []}
    for url in urls:
        print(url)
        image_bytes = base64.b64encode(requests.get(url).content).decode('utf-8')
#        image_bytes = requests.get(url).content
        req['instances'].append({'b64': image_bytes})

    json_response = requests.post('http://localhost:8501/v1/models/rssc/versions/1:predict', 
                                      headers=headers,
                                      data=json.dumps(req))
    return '{}'.format(json.dumps(json_response.json(), indent=4))

