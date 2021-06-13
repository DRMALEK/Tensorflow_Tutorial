import tensorflow as tf

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
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # At every epoch 20% of the neurons's contribuatiion to the next layer is disabled
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.nn.softmax),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Define the optimizer, loss and the metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'mse'])

# Start training
print('Start training')
model.fit(train_X, train_Y, epochs=5)

# Run evalutation
print('Start evaluation')
model.evaluate(test_X, test_Y)