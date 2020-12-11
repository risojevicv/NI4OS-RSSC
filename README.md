# NI4OS-RSSC
NI4OS Remote Sensing Scene Classification

# New

Directory `service` contains code for tensorflow serving model. Directory `webapp` contains code for frontend app that can be used for service demonstration. Webapp also contains code for wrapping tf serving calls.

Run with `docker-compose up`

N.B.: Install the latest stable version of Docker Compose as per instructions on https://docs.docker.com/compose/install/

Service accepts request at port 8080.

Test API with `curl -d "urls=https://drive.etfbl.net/s/Wat52qcpm5P9YzC/preview,https://drive.etfbl.net/s/ZjpLszJDBopfCWk/preview" -X POST http://localhost:8080/url-api` or using `image_classification_jpeg_list.py`.

