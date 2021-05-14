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

# Building the model
# Setting up layers:
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training the model
# Feed the model:
model.fit(train_images, train_labels, epochs=10)

# Evaluating accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:
print('Predictions array: ', predictions[0])

# A prediction is an array of 10 numbers.
# They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing.
# We can see which label has the highest confidence value:
print('Highest value: ', np.argmax(predictions[0]))

# According to the output, the model is most confident that this image is an ankle boot, or class_names[9].
# Examining the test label shows that this classification is correct:
print(test_labels[0])

# Graph this to look at the full set of 10 class predictions.
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100*np.max(predictions_array),
        class_names[true_label]
    ), color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Verify Predictions
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Using the trained model
img = test_images[1]

# tf.keras models are optimized to make predictions on a batch, or collection, of examples at once.
# Accordingly, even though we're using a single image, we need to add it to a list:
img = (np.expand_dims(img,0))

# Now predict the correct label for this image:
predictions_single = probability_model.predict(img)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

# tf.keras.Model.predict returns a list of lists—one list for each image in the batch of data.
# Grab the predictions for our (only) image in the batch:
np.argmax(predictions_single[0])
