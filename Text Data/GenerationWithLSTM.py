import tensorflow as tf
import sys
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Preprocess the data
# In the book the author used 'an excerpt from a traditional Irish song from the 1860s'
# however i could not find it and use the 'The Project Gutenberg' book instead
with open('./irish-lyrics-eof.txt', 'r') as input_file:
    data = input_file.read()

# create sequences from the text
tokenizer = Tokenizer()
corpus = data.lower().split('\n')
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Turning a sequence into a number of input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Perform pre_padding
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(
    input_sequences, maxlen=max_sequence_len, padding='pre'))

# Finally
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]

# One-hot-encoding to the labels
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 8))
# Due to text gernation nature, use embadding_dims equal to the max_word count
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(max_sequence_len-1)))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Start training
history = model.fit(xs, ys, epochs=1000, verbose=1)

# Do some tests and predict the next word
seed_text = "in the town of athy"
token_list = tokenizer.texts_to_sequences([seed_text])[0]
token_list = pad_sequences(
    [token_list], maxlen=max_sequence_len-1, padding='pre')
predicted = np.argmax(model.predict(token_list), axis=-1)
print(predicted)

# Compounding the predicitons to generate text
seed_text = "These reflections have dispelled the agitation with which I began my"
next_words = 10

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break

    seed_text += " " + output_word

# Print the final predicted text !!!
print(seed_text)
