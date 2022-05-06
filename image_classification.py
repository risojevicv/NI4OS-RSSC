#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""

RS image classification test

Created on Wed Mar 25 09:40:42 2020

@author: vlado
"""

import requests
import json
import base64
import time

# test_paths = ['webapp/app/static/images/mediumresidential_58.jpg',
#               'webapp/app/static/images/farmland_52.jpg',
#               'webapp/app/static/images/bridge_22.jpg',
#               'webapp/app/static/images/storagetanks_1.jpg']
test_paths = ['/home/research/vlado/data/AID_jpeg/Airport/airport_17.jpg']

headers = {'content-type': 'application/json'}
req = {'signature_name': 'serving_default', 
       'instances': []}

for image_path in test_paths:
    with open(image_path, 'rb') as f:
        image_bytes = base64.b64encode(f.read()).decode('utf-8')
        req['instances'].append({'b64': image_bytes})

data = json.dumps(req)

t0 = time.time()
json_response = requests.post('http://localhost:8080/upload-api',
                                headers=headers,
                                data=data)

print('Elapsed time: {:.2f}'.format(time.time() - t0))

print(json.dumps(json_response.json(), indent=4))

results = [{classname: pred for classname, pred in zip(res['classnames'], res['probabilities'])} for res in json_response.json()['predictions']]

print(results)
