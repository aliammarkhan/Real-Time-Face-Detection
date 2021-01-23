# Real-Time-Face-Recognition
Recognizing faces of seven main characters of Big Bang theory in real time.

This is a simple face detection project which detects faces of characters from Big Bang Theory.

## Requirements
You need the following packages to run the scripts:
* Tensorflow-2.1.0
* Python-3.7
* simple_image_download
* OpenCv
* Pillow

The above packages can be installed as follows:

### Python

Install python from this link https://www.python.org/downloads/release/python-370/

### Tensorflow-2.1.0

Using Conda

```bash
conda create -n tf tensorflow
conda activate tf
```
Using pip

```bash
pip install tensorflow==2.1.0
```
### simple_image_download

```bash
pip install simple_image_download
```
### OpenCv

```bash
pip install opencv-python
```
### Pillow

```bash
pip install Pillow
```

## How to run

### Preparing the Dataset

First run the script scraping_images.py , this will prepare the dataset by downloading pictures of 7 main characters from google images and cropping their faces. The dataset is arranged according to the format accepted by ImageDataGenerator. However after running this , you have to manually go through the pictures and delete unwanted pictures
(like pictures of other characters , legos , etc).

```bash
python scraping_images.py
```
### Prearing the model

The model model.ipynb was trained and then downloaded from google collab , however you can also run it on your local machine. The convolutional and pooling layers of VGG16
were used.

### Running the main script

The main script is face_detector.py , this will load the saved model and detect faces of characters from an episode.

## Results

The model was trained only on a few pictures(161), for better performace , prepare a dataset with more pictures and adjust the layers of neural network accordingly.

![s1](https://user-images.githubusercontent.com/50051546/91659166-c152de80-eae7-11ea-805d-19d840ca393d.png)

![s2](https://user-images.githubusercontent.com/50051546/91659176-cfa0fa80-eae7-11ea-9527-5e1545b08205.png)


