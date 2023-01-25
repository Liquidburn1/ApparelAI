//Hassan Ali Qadir
//This code creates a convolutional neural network using the TensorFlow library and trains it on the fashion MNIST dataset. 
//The model is then evaluated on a test dataset and the test accuracy is printed.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create a sequential model
model = keras.Sequential()

# Add convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the feature maps
model.add(layers.Flatten())

# Add fully connected layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load the clothing and shoe data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Scale the pixel values to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert labels to categorical format
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
