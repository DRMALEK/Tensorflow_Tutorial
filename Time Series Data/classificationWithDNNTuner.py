import sys
sys.path.append('../')
from kerastuner.tuners import RandomSearch
from matplotlib import pyplot as plt
from Utils.Plot import Plot
from Utils.Series import Series
import numpy as np
import tensorflow as tf

# Craete a simple series
time_seq = np.arange(4 * 365 + 1, dtype="float32")
series = Series.trend(time_seq, 0.1)
baseline = 10
amplitude = 20
slope = 0.09
noise_level = 5

series = baseline + Series.trend(time_seq, slope)
series += Series.seasonality(time_seq, period=365, amplitude=amplitude)
series += Series.noise(time_seq, noise_level, seed=42)

# Create a windowed dataset from the series
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000
dataset = Series.windowed_dataset(
    series, window_size, batch_size, shuffle_buffer_size)


# Main function for building the model
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=10,
                                                 max_value=30, step=2), activation='relu', input_shape=[window_size]))
    model.add(tf.keras.layers.Dense(units=hp.Int(
        'units2', min_value=1, max_value=30, step=2), activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(
        hp.Choice('momentum', values=[.9, .7, .5, .3]), lr=1e-5))
    return model


# Initalize a RandomSearch object
tuner = RandomSearch(build_model,
                     objective='loss',
                     max_trials=150,
                     executions_per_trial=3,
                     directory='my_dir',
                     project_name='hello')

# Start hyperParamter tunning
tuner.search(dataset, epochs=100, verbose=0)

# Show the tunning results
tuner.results_summary()
