# NI4OS-RSSC
## NI4OS Remote Sensing Scene Classification


Directory `service` contains code for TensorFlow Serving model. The trained models can be downloaded  [here](https://drive.google.com/drive/folders/1Yp_B--dWDimvJFLA3cssxTrHTcZkV8Hu?usp=sharing) and should be put to `service/models` directory. Directory `webapp` contains code for frontend app that can be used for service demonstration. Webapp also contains code for wrapping TensorFlow Serving calls.

Run with `docker-compose up`

N.B.: Install the latest stable version of Docker Compose as per instructions on https://docs.docker.com/compose/install/

The service accepts request at port 8080.

Test API with `curl -d "urls=https://drive.etfbl.net/s/Wat52qcpm5P9YzC/preview,https://drive.etfbl.net/s/ZjpLszJDBopfCWk/preview" -X POST http://localhost:8080/url-api` or using `image_classification_jpeg_list.py`.

