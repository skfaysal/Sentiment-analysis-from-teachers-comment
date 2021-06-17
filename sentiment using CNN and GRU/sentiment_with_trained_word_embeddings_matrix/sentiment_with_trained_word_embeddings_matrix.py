"""
This code is for sentiment analysis using CNN,GRU and LSTM using previously created Word Embeddings
"""
print("hi")
# importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
import os
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.initializers import Constant
from keras.layers.embeddings import Embedding




# Importing dataset
df = pd.read_csv(str(os.getcwd())+'/Training Set.csv')
print(df.shape)
# dropping out rows which contains null
df=df.dropna()
print(df.shape)

# Create list of list of words from sentences
review_lines = list()
lines = df['meeting_remarks'].values.tolist()
for line in lines:
    tokens = word_tokenize(line)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words    
    # stop_words = set(stopwords.words('english'))
    # words = [w for w in words if not w in stop_words]
    review_lines.append(words)

print(len(review_lines))
print(review_lines)
j = 0
for i in review_lines:
    print(len(i))
    j += len(i)
print(j)

# Create word embeddings from the list of list of words created
EMBEDDING_DIM = 100
# train word2vec model
model = gensim.models.Word2Vec(sentences=review_lines, size=EMBEDDING_DIM, window=5, workers=4, min_count=1)
# vocab size
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))
print(words)
print(len(words))
# save model in ASCII (word2vec) format
filename = './mentoring_embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

# similar words of improvement using word embeddings
print(model.wv.most_similar('improvement'))  # , topn =1)

# create a dictionary as
# key=words,value=corresponding co-efficients/weights
embeddings_index = {}
f = open(os.path.join('','./mentoring_embedding_word2vec.txt'),encoding="utf-8")
for line in f:
    #print(line)
    values = line.split()
    #print(values)
    word = values[0]
    #print(word)
    coefs = np.asarray(values[1:])
    print(coefs)
    embeddings_index[word] = coefs
f.close()
print(len(embeddings_index))

"""
If the target variable is not encoded
df_enc = pd.get_dummies(df, columns=['Sentiment'],drop_first=True)
x = df_enc['meeting_remarks'].values
y = df_enc.iloc[:, 1:].values
"""

# Splitting
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
print(total_reviews)
a = tokenizer_obj.fit_on_texts(total_reviews)
print(type(a))
# pad sequences
max_length = 100  # try other options like mean

word_index = tokenizer_obj.word_index
print('Found %s unique tokens.' % len(word_index))
# define vocabulary size
# Vocabolary size should be 1 plus lenth of word_index
vocab_size = len(tokenizer_obj.word_index) + 1

X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

print(len(X_test_tokens))
# Here post means assigning zero at end
X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')

print('Shape of X_train_pad tensor:', X_train_pad.shape)
print('Shape of y_train tensor:', y_train.shape)

print('Shape of X_test_pad tensor:', X_test_pad.shape)
print('Shape of y_test tensor:', y_test.shape)

# Create embedding matrix with shape (6303,100)
# where 6303 = unique tockens and 100= number of columns
EMBEDDING_DIM = 100
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print(num_words)
print(embedding_matrix)
embedding_df=pd.DataFrame(embedding_matrix)

"""CNN Model"""
model1 = Sequential()
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_length,
                            trainable=False)

model1.add(embedding_layer)
model1.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model1.add(MaxPooling1D(pool_size=2))
model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))
print(model1.summary())

# compile network
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model1.fit(X_train_pad, y_train, batch_size=128, epochs=2, validation_data=(X_test_pad, y_test), verbose=2)

# evaluate the model
loss, accuracy = model1.evaluate(X_test_pad, y_test, batch_size=128)
print('Accuracy: %f' % (accuracy * 100))

# saving the model
filename = './CNN.pkl'
pickle.dump(model, open(filename, 'wb'))


"""GRU Model"""
model2 = Sequential()
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_length,
                            trainable=False)
model2.add(embedding_layer)
model2.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Summary of the built model...')
print(model2.summary())
print('Train...')
model2.fit(X_train_pad, y_train, batch_size=128, epochs=2, validation_data=(X_test_pad, y_test), verbose=2)
print('Testing...')
score, acc = model2.evaluate(X_test_pad, y_test, batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)
print("Accuracy: {0:.2%}".format(acc))


# saving the model
filename = './GRU.pkl'
pickle.dump(model, open(filename, 'wb'))



# Let us test some  samples
test_sample_1 = "Financial problem and need support"
test_sample_2 = "She needs to be regular and study hard."
test_sample_3 = "Her all dues are clear. She has participated in all the tasks given by her teachers as well."
test_sample_4 = "Have to improved. Result is Not Satisfiable."
test_sample_5 = "Improved but not remarkable"
test_sample_6 = "he is best"
test_sample_7 = "he is improving"
test_sample_8 = "he is irregular"

test_samples = [test_sample_1, test_sample_2, test_sample_3, test_sample_4, test_sample_5,test_sample_6,test_sample_7,
                test_sample_8]

test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)
test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)

# predict
predict = model1.predict(x=test_samples_tokens_pad)
print(predict)

