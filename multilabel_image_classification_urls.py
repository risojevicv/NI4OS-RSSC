#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""

RS image classification test

Created on Wed Mar 25 09:40:42 2020

@author: vlado
"""

import requests
import json
import time

test_urls = [r'https://raw.githubusercontent.com/risojevicv/NI4OS-RSSC/main/webapp/app/static/images/mediumresidential_58.jpg',
             r'https://raw.githubusercontent.com/risojevicv/NI4OS-RSSC/main/webapp/app/static/images/bridge_22.jpg']

headers = {'content-type': 'application/x-www-form-urlencoded'}
data = 'urls=' + ','.join(test_urls)

t0 = time.time()
json_response = requests.post('http://localhost:8080/multilabel-url-api',
                                headers=headers,
                                data=data)
print('Elapsed time: {:.2f}'.format(time.time() - t0))

print(json.dumps(json_response.json(), indent=4))

results = [{classname: pred for classname, pred in zip(res['classnames'], res['probabilities'])} for res in json_response.json()['predictions']]

print(results)

