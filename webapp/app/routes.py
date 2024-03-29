import os
import json
import base64
import requests
from flask import render_template, redirect, url_for, escape, request, flash
from app import app
from app.forms import URLForm, FilesForm
from app.utils import perform_url_request, perform_upload_request

MAX_FILES_ALLOWED = 10
MAX_FILES_ALLOWED_PATCHES = 1

@app.route('/url-api', methods=['POST'])
def url_api():
    urls = request.form['urls'].split(',')
    headers = {'content-type': 'application/json'}
    req = {'signature_name': 'serving_default',
           'instances': []}
    for url in urls:
        image_bytes = base64.b64encode(requests.get(url).content).decode('utf-8')
        req['instances'].append({'b64': image_bytes})

    json_response = requests.post('http://rssc:8501/v1/models/rssc/versions/1:predict', 
                                      headers=headers,
                                      data=json.dumps(req))
    return json_response.json()

@app.route('/multilabel-url-api', methods=['POST'])
def multilabel_url_api():
    urls = request.form['urls'].split(',')
    headers = {'content-type': 'application/json'}
    req = {'signature_name': 'serving_default',
           'instances': []}
    for url in urls:
        image_bytes = base64.b64encode(requests.get(url).content).decode('utf-8')
        req['instances'].append({'b64': image_bytes})

    json_response = requests.post('http://rssc:8501/v1/models/multilabel-rssc/versions/1:predict', 
                                      headers=headers,
                                      data=json.dumps(req))
    return json_response.json()

@app.route('/upload-api', methods=['POST'])
def classify_images():
    json_response = requests.post('http://rssc:8501/v1/models/rssc/versions/1:predict',
            headers=request.headers,
            data=request.data)
    return json_response.json()

@app.route('/multilabel-upload-api', methods=['POST'])
def classify_images_multilabel():
    json_response = requests.post('http://rssc:8501/v1/models/multilabel-rssc/versions/1:predict',
            headers=request.headers,
            data=request.data)
    return json_response.json()

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    return redirect(url_for('upload'))

@app.route('/url', methods=['GET', 'POST'])
def url():
    form = URLForm()

    if form.validate_on_submit():
        result = perform_url_request(form.url.data, form.task.data)

        return render_template('result.html', title='Results', res=result, task=form.task.data.lower())

    return render_template('url.html', title='URL', form=form)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form_upload = FilesForm()

    examples = ['farmland_52.jpg', 'mediumresidential_58.jpg', 'bridge_22.jpg', 'storagetanks_1.jpg']

    if form_upload.validate_on_submit():
        max_files = MAX_FILES_ALLOWED_PATCHES if form_upload.task.data.lower().startswith('patch') else MAX_FILES_ALLOWED
        if len(form_upload.files.data) > max_files:
            flash('Maximum of {:d} files allowed!'.format(max_files))
        result, labeled, colors = perform_upload_request(form_upload.files.data[:max_files], form_upload.task.data)

        return render_template('result.html', title='Results',
                               res=result, labeled=labeled, colors=colors, task=form_upload.task.data.lower())

    return render_template('files.html', title='Upload', 
            form=form_upload, examples=examples)


@app.route('/example', methods=['POST'])
def example():
    form_upload = FilesForm()

    examples = ['farmland_52.jpg', 'mediumresidential_58.jpg', 'bridge_22.jpg', 'storagetanks_1.jpg']

    ex = request.form.get('example')
    if ex in examples:
        image_path = os.path.join('static/images', ex)
        with app.open_resource(image_path, 'rb') as f:
            f.mimetype = 'image/jpeg'
            result, labeled, colors = perform_upload_request([f, ], form_upload.task.data.lower())

            return render_template('result.html', title='Results',
                                    res=result, labeled=labeled, colors=colors, task=form_upload.task.data.lower())
    else:
        return render_template('files.html', title='Upload', 
                form=form_upload, examples=examples)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', title='404 - Not Found'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html', title='500 - Server Error'), 500
