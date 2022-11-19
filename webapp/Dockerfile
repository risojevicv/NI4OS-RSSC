FROM tiangolo/uwsgi-nginx-flask:python3.8
ENV STATIC_URL /static
ENV STATIC_PATH /app/app/static
ENV SECRET_KEY 6c316640c2938e72b34f236e4a1d137a905396bc9e94aa40d3e73a1e795ed73b5ab92b15aac4e16ff3cb7fe771839d2162f6
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
RUN echo "uwsgi_read_timeout 300s;" > /etc/nginx/conf.d/custom_timeout.conf
COPY . /app