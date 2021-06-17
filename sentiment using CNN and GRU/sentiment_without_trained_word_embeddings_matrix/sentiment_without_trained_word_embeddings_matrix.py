"""
This code is for sentiment analysis using CNN and GRU without previously created Word Embeddings Matrix
This procedure performs better for our dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU,Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.initializers import Constant

df = pd.read_csv('./Training Set.csv')

print(df.head(10))
df.columns = ['meeting_remarks', 'Sentiment']
print(df.columns)
# drop rows which contains null value
df = df.dropna()

# get number of labels of target variable
labels = df['Sentiment'].unique()

print(labels)
# count of each label of target variable
print(df.groupby('Sentiment').size())

# encode target variable
df_enc = pd.get_dummies(df, columns=['Sentiment'],drop_first=True)

# Splitiing
x = df['meeting_remarks'].values
y = df.iloc[:, 1:].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# summarize size
print("Training data: ")
print(x.shape)
print(y.shape)

# Tockenizing and padding
# Tockenizing refers to creating unique words from list of sentences
# Padding refers to creating same dimensional data for all by assigning zero at end

tokenizer_obj = Tokenizer()
total_reviews = df['meeting_remarks'].values
# Tokenize
a = tokenizer_obj.fit_on_texts(total_reviews)

# pad sequences
max_length = 80  # try other options like mean
# define vocabulary size
vocab_size = len(tokenizer_obj.word_index) + 1
print(vocab_size)

# final tokens
X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

print(len(X_test_tokens))
# Padded tokens
X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')

print('Shape of X_train_pad tensor:', X_train_pad.shape)
print('Shape of y_train tensor:', y_train.shape)
print('Shape of X_test_pad tensor:', X_test_pad.shape)
print('Shape of y_test tensor:', y_test.shape)

"""Gated Recurrent Unit(GRU) Model"""
EMBEDDING_DIM = 100
print('Build model...')
model1 = Sequential()
model1.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
model1.add(GRU(units=128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model1.add(GRU(units=64, dropout=0.2, recurrent_dropout=0.2,return_sequences=False))
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(64, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(1, activation='sigmoid'))

# Compiling
model1.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.005), metrics=['accuracy'])

print('Summary of the built model...')
print(model1.summary())

print('Train.....')
model1.fit(X_train_pad, y_train, batch_size=64, epochs=3, validation_data=(X_test_pad, y_test), verbose=2)

print('Testing...')
score, acc = model1.evaluate(X_test_pad, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
print("Accuracy: {0:.2%}".format(acc))

# saving the model
filename = './GRU.pkl'
pickle.dump(model, open(filename, 'wb'))


"""Convolutional 1D(CONV 1D) Model"""
EMBEDDING_DIM = 100

# define model
model2 = Sequential()
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
model2.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
model2.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model2.add(MaxPooling1D(pool_size=2))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
print(model2.summary())

# compile network
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model2.fit(X_train_pad, y_train, batch_size=128, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)
print('Testing...')
score, acc = model2.evaluate(X_test_pad, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
print("Accuracy: {0:.2%}".format(acc))


# saving the model
filename ='./CNN.pkl'
pickle.dump(model, open(filename, 'wb'))


"""Testing on new data"""
test_sample_1 = "Financial problem and need support"
test_sample_2 = "She needs to be regular and study hard."
test_sample_3 = "Her all dues are clear. She has participated in all the tasks given by her teachers as well."
test_sample_4 = "Have to improved. Result is Not Satisfiable."
test_sample_5 = "Improved but not remarkable"
test_sample_6= "he is regular"
test_sample_7= "he killed someone"

test_samples = [test_sample_1, test_sample_2, test_sample_3, test_sample_4, test_sample_5,test_sample_6,test_sample_7]

test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)
test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)

# predict
predict = model1.predict(x=test_samples_tokens_pad)
print(predict)
