from tensorflow.keras.applications.inception_v3 import InceptionV3
import urllib.request
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

# Download the weights
weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

# Create the model
pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=weights_url
)

# Stop all the layers from training
for layer in pre_trained_model.layers:
    layer.trainable = False

# choose a layer (You can choose any, by taking a look at model.summary())
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Build the new dense layers
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activaiton='relu')(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Combine the pretrained model with new created dense layers
model = tf.keras.Model(pre_trained_model, x)
model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
