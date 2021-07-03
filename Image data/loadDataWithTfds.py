import tensorflow as tf
import tensorflow_datasets as tfds

# load the data with tfds ((( Make sure to install tensorflow-datasets package )))
(training_images, training_labels), (test_images, test_labels) = tfds.as_numpy(
    tfds.load('fashion_mnist', split=['train', 'test'], batch_size=-1, as_supervised=True))

# build the model
model = tf.keras.models.Sequential([
    # one, since the tfds out has 3 dimensions
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(training_images, training_labels, epochs=5)
