# new-plant-diseases
## Image Classification

## Getting Data

The New Plant Diseases Dataset can be acquired [here](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

To generate features and labels, use the `ImageDataGenerator` from Keras:

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=None,
    shear_range=0.3,
    zoom_range=0.5,
    horizontal_flip=True
)
