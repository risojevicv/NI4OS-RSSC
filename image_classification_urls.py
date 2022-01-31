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

test_urls = [r'https://drive.etfbl.net/s/Wat52qcpm5P9YzC/preview',
             r'https://drive.etfbl.net/s/ZjpLszJDBopfCWk/preview']

headers = {'content-type': 'application/x-www-form-urlencoded'}
data = 'urls=' + ','.join(test_urls)

t0 = time.time()
json_response = requests.post('http://localhost:8080/url-api',
                                headers=headers,
                                data=data)
print('Elapsed time: {:.2f}'.format(time.time() - t0))

print(json.dumps(json_response.json(), indent=4))

results = [{classname: pred for classname, pred in zip(res['classnames'], res['preds'])} for res in json_response.json()['predictions']]

print(results)

