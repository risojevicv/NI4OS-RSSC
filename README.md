# Web Service for Remote Sensing Scene Classification (RS2C)

## Background

The number of remote sensing platforms for monitoring the surface of the Earth has been growing at ever increasing pace. Consequently, the amount of remote sensing images has surpassed our abilities for manual analysis. Luckily, recent advances in computer vision have provided us with new tools for automatic analysis of remote sensing images. 

Scene classification is a key step in remote sensing image analysis aiming to annotate each image with labels from a pre-defined set. These annotations can be used for applications like land cover and land use classification, monitoring urban growth, monitoring and forecasting climate changes, to name only a few. Recently, using remote sensing for monitoring of ecosystems, insects and animals have also gained in significance. 

Convolutional neural networks (CNNs) have become a de facto standard in various computer vision tasks, ranging from image classification to object detection to semantic segmentation. However, training of CNNs requires considerable computing resources as well as large labeled training sets. In order to lift the burden of CNN training and enable end-users to reap the benefits of using powerful CNN-based classifiers, RS2C offers access to pre-trained CNN models for remote sensing scene classification. In other words, RS2C aims to assist in making sense of the remote sensing imagery acquired using various sensing platforms, ranging from satellites to UAVs. The default problem in remote sensing scene classification is single-label multi-class classification. However, in many cases remote sensing images cannot be accurately described using a single label. For that reason, RS2C also features multi-label classification. Main user communities envisaged to benefit from this service are in the areas of agriculture, food production, urban planning, and environment protection, but practitioners in other fields may also find it useful. 

## Service Description

RS2C is a RESTful web service and web application for remote sensing scene classification based on convolutional neural networks. Currently, ResNet-50 [1] pre-trained on ImageNet and fine-tuned on [MLRSNet](https://github.com/cugbrs/MLRSNet) [2] is used for classification. The web service is implemented in Python using [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) and [Flask](https://flask.palletsprojects.com/en/2.0.x/). The RS2C API provides two groups of methods:

+ Single-label classification - each image is classified into one of the following 46 mutually exclusive scene categories: airplane, airport, bareland, baseball diamond, basketball court, beach, bridge, chaparral, cloud, commercial area, dense residential area, desert, eroded farmland, farmland, forest, freeway, golf course, ground track field, harbor & port, industrial area, intersection, island, lake, meadow, mobile home park, mountain, overpass, park, parking lot, parkway, railway, railway station, river, roundabout, shipping yard, snowberg, sparse residential area, stadium, storage tank, swimmimg pool, tennis court, terrace, transmission tower, vegetable greenhouse, wetland, wind turbine.

  - `url-api`: POST method accepting a comma-separated list of JPEG image URLs.
  
  - `upload-api`: POST method accepting the list of JPEG images.

  Both methods return a JSON object with a list of categories and corresponding confidences for each image.

+ Multi-label classification (tagging) - each image is assigned multiple labels from the list: airplane, freeway, roundabout,,airport,golf course, runway, bare soil, grass, sand, baseball diamond, greenhouse, sea, basketball court, gully, ships, beach, habor, snow, bridge, intersection, snowberg, buildings, island, sparse residential area, cars, lake, stadium, chaparral, mobile home, swimming pool, cloud, mountain, tanks, containers, overpass, tennis court, crosswalk, park, terrace, dense residential area, parking lot, track, desert, parkway, trail, dock, pavement, transmission tower, factory, railway, trees, field, railway station, water, football field, river, wetland, forest, road, wind turbine.

    - `multilabel-url-api`: POST method accepting a comma-separated list of JPEG image URLs.
  
    - `multilabel-upload-api`: POST method accepting the list of JPEG images.

Both methods return a JSON object with a list of labels and confidences that each label can be assigned to the image.

You can try the service at https://rs2c.etfbl.net.

For training/fine-tuning the models please refer to [this](https://github.com/risojevicv/RSSC-transfer) repository. 

## Installation and Running

For running the service we use Docker and Docker Compose. Please install the latest stable versions of Docker Engine and Docker Compose as per  instructions on https://docs.docker.com/compose/install/.

1. Pull the Docker images with `docker-compose pull`,
2. Run the service with `docker-compose up`.

If you wish to build the service yourself, directory `service` contains code for building a Docker image with the TensorFlow Serving model. The trained models can be downloaded  [here](https://drive.google.com/drive/folders/1Yp_B--dWDimvJFLA3cssxTrHTcZkV8Hu?usp=sharing) and should be put into `service/models` directory. 

Directory `webapp` contains code for building a Docker image with the frontend app that can be used for service demonstration as well as the code for wrapping TensorFlow Serving calls. The index files for k-nearest neighbor-based out-of-distribution detection can be downloaded [here](https://drive.google.com/drive/folders/1NGJjlWclp5bJAvWqY0_zB8l0JAO8nDY9?usp=sharing) and should be put into `webapp/app/knn_indices' directory.

Finally, run the service with `docker-compose up`. 

The service accepts request at port 8080. You can test the API for single-label classification with:

`curl -d "urls=https://raw.githubusercontent.com/risojevicv/NI4OS-RSSC/main/webapp/app/static/images/mediumresidential_58.jpg, https://raw.githubusercontent.com/risojevicv/NI4OS-RSSC/main/webapp/app/static/images/bridge_22.jpg" http:/localhost:8080/url-api`.

To perform multi-label classification of the same images, you can use:

`curl -d "urls=https://raw.githubusercontent.com/risojevicv/NI4OS-RSSC/main/webapp/app/static/images/mediumresidential_58.jpg, https://raw.githubusercontent.com/risojevicv/NI4OS-RSSC/main/webapp/app/static/images/bridge_22.jpg" http://localhost:8080/multilabel-url-api`

Alternatively, the service can be consumed programmatically. The examples in Python are given in [image_classification.py](https://github.com/risojevicv/NI4OS-RSSC/blob/main/image_classification.py) and [image_classification_urls.py](https://github.com/risojevicv/NI4OS-RSSC/blob/main/image_classification_urls.py).


## References

[1] He, Kaiming, et al. "Deep residual learning for image recognition",
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770-778, 2016.

[2] Qi, Xiaoman, et al. "MLRSNet: A multi-label high spatial resolution remote sensing dataset for semantic scene understanding.", ISPRS Journal of Photogrammetry and Remote Sensing 169 (2020): 337-350.
