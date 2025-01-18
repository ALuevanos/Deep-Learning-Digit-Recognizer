# Handwritten digit recognition for MNIST dataset using Convolutional Neural Networks
# Step 1: Import all required keras libraries
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Step 2: Load and return training and test datasets
def load_dataset():
    # 2a. Load dataset X_train, X_test, y_train, y_test via imported keras library
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # 2b. reshape for X train and test vars
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
    # 2c. normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # 2d. Convert y_train and y_test to categorical classes
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # 2e. return your X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test

# Step 3: Load your saved model
your_model_name = load_model('digitRecognizer.h5')

# Step 4: Evaluate your model
X_train, X_test, y_train, y_test = load_dataset()
evaluation_results = your_model_name.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", evaluation_results[0])
print("Test Accuracy:", evaluation_results[1])

# Code below to make a prediction for a new image.
# Step 5: This section below is optional and can be copied from your digitRecognizer.py file from Step 8 onwards
# - load required keras libraries

# Step 6: load and normalize new image
def load_new_image(path):
    # 6a. load new image
    newImage = load_img(path, color_mode='grayscale', target_size=(28, 28))
    # 6b. Convert image to array
    newImage = img_to_array(newImage)
    # 6c. reshape into a single sample with 1 channel
    newImage = newImage.reshape(1, 28, 28, 1).astype('float32')
    # 6d. normalize image data
    newImage = newImage / 255
    # 6e. return newImage
    return newImage

# Step 7: load a new image and predict its class
def test_model_performance():
    # 7a. Call the above load image function
    img = load_new_image('sample_images/digit1.png')
    # 7b. load your CNN model (digitRecognizer.h5 file)
    #your_model_name = load_model('digitRecognizer.h5')
    # 7c. predict the class
    predictions = your_model_name.predict(img)
    imageClass = np.argmax(predictions, axis=-1)
    # 7d. Print prediction result
    print("Predicted class:", imageClass[0])

# Step 8: Test model performance
test_model_performance()
