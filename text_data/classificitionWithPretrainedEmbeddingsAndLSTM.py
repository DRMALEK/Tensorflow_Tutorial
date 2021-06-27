import numpy as np
import tensorflow as tf
import urllib.request
import zipfile

# Download the trained embadding
url = "https://ndownloader.figshare.com/files/21119196"
download_dir = "tmp/glove.txt.gz"
urllib.request.urlretrieve(url, download_dir)

zip_ref = zipfile.ZipFile(download_dir, 'r') 
zip_ref.extractall('/tmp/glove') 
zip_ref.close()

# Build a word: dimensional coefficients dictoinary
glove_embeddings = dict() 
f = open('/tmp/glove/glove.twitter.27B.25d.txt') 
for line in f: 
    values = line.split() 
    word = values[0] 
    coefs = np.asarray(values[1:], dtype='float32') 
    glove_embeddings[word] = coefs


embedding_dim = 25 # Same as the pretrained model dimension
vocab_size = 13200 # Based on some kind of analysis (please refer to the book p:141-143 to understand how we choose this number)

# Create an embedding matrix using the vocab from the dataset 
# and embeddings from the pre_trained model
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in tokenizer.word_index.items():
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

# Build the model
model = tf.keras.Sequential([ tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False),
                              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
                              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)), 
                              tf.keras.layers.Dense(24, activation='relu'), 
                              tf.keras.layers.Dense(1, activation='sigmoid')
])

adam = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False) 
history = model.compile(loss='binary_crossentropy',optimizer=adam, metrics=['accuracy'])

train_loss = history.history['loss']
train_acc  = history.history['accuracy']
val_loss   = history.history['val_loss']
val_acc    = history.history['val_accuracy']

# Plot the train vs val accuracy chart
Plot.plot_history_line('Epochs', 'Accuracy', train_acc, 'accuracy', val_acc, 'val_accuracy', figure_name='PretrainedEmbdAndLSTMAcc.png')

# Plot the train vs val loss chart
Plot.plot_history_line('Epochs', 'Loss', train_loss, 'loss', val_loss, 'val_loss', figure_name='PretrainedEmbdAndLSTMLoss.png')