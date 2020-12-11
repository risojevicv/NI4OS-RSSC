# NI4OS-RSSC
NI4OS Remote Sensing Scene Classification

# New

Directory `service` contains code for tensorflow serving model. Directory `webapp` contains code for frontend app that can be used for service demonstration. Webapp also contains code for wrapping tf serving calls.

Run with `docker-compuse up`

# Old

Code for building a Tensorflow SavedModel for remote sensing scene classification. The model accepts JPEG images and classifies it into land-use/land-cover classes.

Directory `finetuning` contains code for finetuning an ImageNet pretrained model on a dataset of remote sensing images.

Run TF model server, run `wrapper.py` in Flask

Test with `curl -d "urls=https://drive.etfbl.net/s/Wat52qcpm5P9YzC/preview,https://drive.etfbl.net/s/ZjpLszJDBopfCWk/preview" -X POST http://localhost:5000`.

