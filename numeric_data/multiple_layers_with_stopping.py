import tensorflow as tf

# Define a callback class, with callback functions
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.95):
            print('\n Reached 95% accuracy so cancelling training!')
            self.model.stop_training = True

callbacks = myCallback()

# Load the dataset
data = tf.keras.datasets.fashion_mnist
(train_X, train_Y),(test_X, test_Y) = data.load_data()

# See the shape of the data
print('train X shape: ', train_X.shape)
print('train Y shape: ', train_Y.shape)
print('test X shape: ', test_X.shape)
print('test Y shape: ', test_Y.shape)

# Normalize the data
train_X = train_X / 255.0
test_X = test_X / 255.0

# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Define the optimizer, loss and the metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Start training with stopping callback
print('Start training')
model.fit(train_X, train_Y, epochs=50, callbacks=[callbacks])

# Run evalutation
print('Start evaluation')
model.evaluate(test_X, test_Y)