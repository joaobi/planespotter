# Plane Spotter
Demo scenario for an App to detect planes in pictures and classify which airline they belong to.

Currently supports:
1. Emirates
2. Singapore Airlines
3. ANA
4. Asiana
5. Korean Airlines
6. Qantas

Key use-case scenarios:
* [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) - The object (plane) detection API used. Used the standard MSCOCO image library
* Custom airline classifier model using Kreas - included in code (images not included)

The scenario includes the code for both client API and web/mobile clients.  



## Overview
This demo includes:
1. apps - mobile, web anc command line apps (both online and offline). Online talks to web_svc. Offline is self-contained.
2. libs - shared libs (TF Object Detection API and PlaneSpotter core lib)
3. models - object (plane) detection and airline classification models used by apps and API
4. web_svc - OpenAPI webservice that detect/predicts plane/airline images

## Getting Started

### Prerequisites
```
numpy
pandas
matplotlib
Pillow
flask
tensorflow==1.8.0
```

### Setup
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

#### Option 1: Docker Image
```
docker run --rm -it planespotter:latest

```

#### Option 2: Deploy Directly
1. Install the pre-requirements from the requirements.txt
2. Run the back-end and/or any client you want

## Authors

* **Joao Bilhim (JB)** - *Initial work*


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used

