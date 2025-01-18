# Handwritten digit recognition for MNIST dataset using Convolutional Neural Networks
# Step 1: Import all required keras libraries
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Step 3: define your CNN model here in this function and then later use this function to create your model
def digit_recognition_cnn():
    # 3a. create your CNN model here with Conv + ReLU + Flatten + Dense layers
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    # 3b. Compile your model with categorical_crossentropy (loss), adam optimizer and accuracy as a metric
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 3c. return your model
    return model

# Step 4: Call digit_recognition_cnn() to build your model
model = digit_recognition_cnn()

# Step 5: Train your model and see the result in Command window. Set epochs to a number between 10 - 20 and batch_size between 150 - 200
X_train, X_test, y_train, y_test = load_dataset()
datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    shear_range=0.3,
    height_shift_range=0.08,
    zoom_range=0.08
)
# Train model
model.fit(datagen.flow(X_train, y_train, batch_size=150), epochs=10, validation_data=(X_test, y_test), verbose=1)


# Step 6: Evaluate your model via your_model_name.evaluate() function and copy the result in your report
model.evaluate(X_test, y_test, verbose=0)

# Step 7: Save your model via your_model_name.save('digitRecognizer.h5')
model.save('digitRecognizer.h5')

# Code below to make a prediction for a new image.

# Step 8: load required keras libraries
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Step 9: load and normalize new image
def load_new_image(path):
    # 9a. load new image
    newImage = load_img(path, color_mode='grayscale', target_size=(28, 28))
    # 9b. Convert image to array
    newImage = img_to_array(newImage)
    # 9c. reshape into a single sample with 1 channel (similar to how you reshaped in load_dataset function)
    newImage = newImage.reshape(1, 28, 28, 1).astype('float32')
    # 9d. normalize image data - Hint: newImage = newImage / 255
    newImage = newImage / 255
    # 9e. return newImage
    return newImage

# Step 10: load a new image and predict its class
def test_model_performance():
    # 10a. Call the above load image function
    cnn_model = load_model('digitRecognizer.h5')
    for x in range(1, 10):
        img_path = f'sample_images/digit{x}.png'
        img = load_new_image(img_path)
        imageClass = cnn_model.predict(img)
        
        # 10b. load your CNN model (digitRecognizer.h5 file)
        predicted_digit = np.argmax(imageClass)

        # 10c. predict the class - Hint: imageClass = your_model_name.predict_classes(img)
        # the imageClass = cnn_model.predict(img)is above insead of being here
        # 10d. Print prediction result
        print(f"digit{x}.png: Predicted digit is {predicted_digit}")

 #Step 11: Test model performance here by calling the above test_model_performance function
test_model_performance()
