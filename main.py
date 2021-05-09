import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# We can import and load the fashion mnist dataset directly from tensorflow.
fashion_mnist = tf.keras.datasets.fashion_mnist

# Loading the dataset returns four NumPy arrays:
# train_images and train_labels arrays are the data the model uses to learn.
# And the model is tested against the test set, the test_images and test_labels arrays.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# The Images are 28x28 NumPy arrays with pixel values ranging from 0 to 255.
# Labels are an array of integers, ranging from 0 to 9. Corresponds to the class of the clothing.

# Since the class names are not included with the dataset, store them here to use later when plotting the images:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Print format of the dataset:
# Following shows there are 60,000 images in the training set, with each image represented as 28x28 pixels.
print('Training dataset shape: ', train_images.shape)

# And there are 60,000 labels in training set:
print('Train labels length: ', len(train_labels))

# Same controls for test dataset
print('Test dataset shape: ', test_images.shape)
print('Test labels length: ', len(test_labels))

# Preprocessing the data
# We can see that the pixel values fall in the range of 0 to 255:
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Scale these values to a range of 0 to 1 before feeding them into the neural network model.
# To do so, we can divide the values by 255. It is important that the trainin set and the testing set be preprocessed in the same way:
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first 25 images to check if everything is correct.
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()













