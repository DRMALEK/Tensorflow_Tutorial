import sys
sys.path.append('../')

from matplotlib import pyplot as plt
from Utils.Plot import Plot
from Utils.Series import Series
import numpy as np
import tensorflow as tf
import sys

# A function to read giss data from a .csv file
def get_data():
    data_file = "./station.csv"
    f = open(data_file)
    data = f.read()
    f.close()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    temperatures = []
    # Loop over the data, and make it as a series of points
    for line in lines:
        if line:
            linedata = line.split(',')
            linedata = linedata[1:13]
            for item in linedata:
                if item:
                    temperatures.append(float(item))
    series = np.asarray(temperatures)
    time = np.arange(len(temperatures), dtype="float32")
    return time, series

# Normalize the data
time, series = get_data()
split_time = 792
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

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