# LeafDiseaseVision Documentation

## Overview
LeafDiseaseVision is a project focused on training a deep-learning model to identify leaf diseases from images. This project leverages TensorFlow and Keras libraries to preprocess image data, build a deep Convolutional Neural Network (CNN), and train the model on the [new plant disease dataset](https://link.springer.com/article/10.1007/s11063-022-10880-z). Serve the model as an endpoint for a web application. 

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [References](#references)

## Requirements
To run this project, you need to install the following libraries:

- Python 3.7+
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

You can install the required libraries using pip:

```bash
pip install tensorflow keras numpy pandas matplotlib
```

## Data Preparation
The New Plant Diseases Dataset can be acquired [here](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
The dataset used in this project is extracted from a ZIP file. The data is organized into training and validation directories.

The dataset is expected to be in the following structure:

```scss
/content/extracted_folder/
    └── New Plant Diseases Dataset(Augmented)/
        ├── train/
        └── valid/
```

```bash
!unzip "/content/drive/MyDrive/Colab Notebooks/archive.zip" -d "/content/extracted_folder"
```

To generate features and labels, use the `ImageDataGenerator` from Keras:

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=None,
    shear_range=0.3,
    zoom_range=0.5,
    horizontal_flip=True
)

train = train_datagen.flow_from_directory(
    directory='/extracted_path/train'
, target_size=(HEIGHT,WIDTH), batch_size=size)

Repeat steps for the validation set
```
## Model Architecture
A deep convolutional network is built for image classification. The model includes layers for image preprocessing, convolution, pooling, and dense layers for classification.

## Training
The model is trained using the ModelCheckpoint and EarlyStopping callbacks to save the best model and to stop training when the validation loss stops improving.

```python
checkpoint = ModelCheckpoint("best_model", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_format='tf')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
callbacks_list = [checkpoint, early_stopping]

history = model.fit(
    train,  # Training data generator
    validation_data=valid,  # Validation data generator
    epochs=epochs,  # Number of epochs for training
    callbacks=[early_stopping]
)
```

The ImageGenerator one hot encodes the labels, the model was compiled- Keras
- optimizer='adam'
- loss='categorical_crossentropy'
-metrics=['accuracy', ..]

## Evaluation
The test dataset follows the same process in the [Data Preparation](#data-preparation) section
 
```python
score = model.evaluate(test)
print(f"Test Loss: {score[0]}")
print(f"Test Accuracy: {score[1]}")
```

## Usage
The model can be utilized in two ways:

### Load model
To use the trained model to predict plant diseases from new images, load the model and preprocess the input images similarly to the training images.

```python
from keras.models import load_model

model = load_model('best_model')

def predict_image(image_path):
    img = load_img(image_path, target_size=(HEIGHT, WIDTH))
    img_array = img_to_array(img)
    prediction = model.predict(img_array)
    return decode_predictions(prediction, top=3)[0]
```

### Web app

Classify your plant image [here] (https://plantapp-4px6bmbdbq-uc.a.run.app/) 

## References
- [Colab Notebook](https://github.com/igbodani/new-plant-diseases/blob/main/PlantVision.ipynb)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
