import sys
sys.path.append('../')

from matplotlib import pyplot as plt
from Utils.Plot import Plot
from Utils.Series import Series
import numpy as np
import tensorflow as tf
import sys

# Get the data
time, series = Series.get_data('station.csv')

# Create a windowed dataset
window_size = 24
batch_size = 12
shuffle_buffer_size = 48
dataset = Series.windowed_dataset(
    x_train, window_size, batch_size, shuffle_buffer_size, expand_dim=True)
valid_dataset = Series.windowed_dataset(
    x_valid, window_size, batch_size, shuffle_buffer_size, expand_dim=True)

# Build the model using Stacked rnn
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(100, return_sequences=True, input_shape=[
                              None, 1], dropout=0.1),
    tf.keras.layers.SimpleRNN(100, dropout=0.1),
    tf.keras.layers.Dense(1)
])

# Fit the model
optimizer = tf.keras.optimizers.SGD(lr=1.5e-6, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer, metrics=["mae"])
model.fit(dataset, epochs=150, verbose=1,
                    validation_data=valid_dataset)