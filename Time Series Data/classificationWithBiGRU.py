import sys
sys.path.append('../')

from matplotlib import pyplot as plt
from Utils.Plot import Plot
from Utils.Series import Series
import numpy as np
import tensorflow as tf
import sys

# Split the data
time, series = Series.get_data('./station_larger.csv')
split_time = 1017 # 75% for train, 25% for validiaton
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# Create a windowed dataset
window_size = 48  # old value was 24
batch_size = 12
shuffle_buffer_size = 48
dataset = Series.windowed_dataset(
    x_train, window_size, batch_size, shuffle_buffer_size, expand_dim=True)
valid_dataset = Series.windowed_dataset(
    x_valid, window_size, batch_size, shuffle_buffer_size, expand_dim=True)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(100,
                            input_shape=[
                                None, 1],
                            return_sequences=True,
                            dropout=0.1,
                            recurrent_dropout=0.1)),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(100,
                            dropout=0.1,
                            recurrent_dropout=0.1)),
    tf.keras.layers.Dense(1),
])

# Fit the model
optimizer = tf.keras.optimizers.SGD(lr=1.5e-6, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer, metrics=["mae"])
history = model.fit(dataset, epochs=50, verbose=1,
                    validation_data=valid_dataset)

# Do evaluation over a set of values
forecast = []
forecast.append(model.predict(valid_dataset))
results = np.array(forecast)
x_valid = series[split_time:]

# Plot the graphs (fix the figure problem)
plt.figure(figsize=(10, 6))
Plot.plot_history_line('Time', 'Value', results, 'forecast',
                       x_valid, 'ground_truth', figure_name='classificationWithBiGRU.png')
