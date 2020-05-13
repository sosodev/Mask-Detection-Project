# Mask Detection Project

A deep learning project to recognize and highlight masked/unmasked human faces using the [Mask RCNN](https://github.com/matterport/Mask_RCNN) architecture on Python 3, Keras and TensorFlow and [Real World Masked Face Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset).

## [Download Trained Model Weights](https://f000.backblazeb2.com/file/cs497-datasets/mask_rcnn_masked_faces.h5)

## Running The Server

* Install [Docker](https://docs.docker.com/get-docker/)
* Build with `docker build . -t detection-server`
* Run with `docker run -p 5000:5000 --rm detection-server`

## Contributing

* Create a fork (or branch if you're a contributor) of the project
* Open the [notebook](https://colab.research.google.com/github/sosodev/Mask-Detection-Project/blob/master/mask_detection.ipynb) in Google Colab
* Make changes
* Choose file -> save copy in GitHub and save to your fork/branch
* Open a pull request! :tada:
