import csv
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import string
import json
import tensorflow as tf
import numpy as np
import sys

# Change this path according to your own system
sys.path.append('/home/malek/Downloads/tensorflow_certifacate_prep-20210616T171606Z-001/tensorflow_certifacate_prep')

from utils.Plot import Plot

# Data preparation
with open('./sarcasm_headlines.json', 'r') as f:
    datastore = json.load(f)
sentences = []
labels = []
table = str.maketrans('', '', string.punctuation)

for item in datastore:
    sentence = item['headline'].lower()
    sentence = sentence.replace(',', ' , ')
    sentence = sentence.replace('.', ' . ')
    sentence = sentence.replace('-', ' - ')
    sentence = sentence.replace('/', ' / ')
    words = sentence.split()
    filtered_sentence = ''
    for word in words:
        word = word.translate(table)              # remove puncs marks
        if word not in stopwords.words('english'):
            filtered_sentence = filtered_sentence + word + ' '     # remove stopwords
    sentences.append(filtered_sentence)
    labels.append(item['is_sarcastic'])

# create training and test datasets
training_size = 20000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# do tokenizing and padding for the traing and the validaiton data
vocab_size = 10000
max_length = 10
trunc_type='post'
padding_type= 'post'
oov_tok = '<OOV>'
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Build the model
model = tf.keras.Sequential([ 
    tf.keras.layers.Embedding(vocab_size, 16),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16), return_sequences=True, dropout=0.2), # Rule of thump : embedding dim == LSTM units count
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16), dropout=0.2), 
    tf.keras.layers.Dense(24, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

# start the training
history = model.fit(x=np.array(training_padded), 
          y=np.array(training_labels), 
          validation_data=(np.array(testing_padded), np.array(testing_labels)),
          epochs=30)

print(history.history.keys())

train_loss = history.history['loss']
train_acc  = history.history['accuracy']
val_loss   = history.history['val_loss']
val_acc    = history.history['val_accuracy']

# Plot the train vs val accuracy chart
Plot.plot_history_line('Epochs', 'Accuracy', train_acc, 'accuracy', val_acc, 'val_accuracy', figure_name='stackedLastmAccuracy.png')

# Plot the train vs val loss chart
Plot.plot_history_line('Epochs', 'Loss', train_loss, 'loss', val_loss, 'val_loss', figure_name='stackedLstmLoss.png')