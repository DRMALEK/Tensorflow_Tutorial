import csv
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import string
import json
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

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
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

hub_layer = hub.KerasLayer(
    # Use a pretrained Embedding
    "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
    output_shape=[20],
    input_shape=[],
    dtype=tf.string,
    trainable=False
)

# Build the model using Embedding layers (10000 voab size, 16 dimesion)
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x=np.array(training_padded), y=np.array(training_labels),
          validation_data=(np.array(testing_padded), np.array(testing_labels)))


# Using the model to predict a list of sentences
test_sentences = ['granny starting to fear spiders in the garden might be real',
                  'game of thrones season finale showing this sunday night',
                  'TensorFlow book will be a best seller']

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(
    test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
