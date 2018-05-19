import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import cPickle as pkl

numpy.random.seed(1)

# Using only top 10k words instead of all
top_words = 10000
skip_top = 20
vocab_size = top_words - skip_top
index_from = 3  # word index offset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words, skip_top=skip_top, index_from=index_from)

#Pad smaller sequences
max_review_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

word_to_id = imdb.get_word_index()
word_to_id = {key: (val + index_from) for key, val in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

# Word Embeddings vector length
embed_length = 100

# Read all embeddings
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    vals = line.split()
    word = vals[0]
    coeffs = numpy.asarray(vals[1:], dtype='float32')
    embeddings_index[word] = coeffs
f.close()

print('Read %s word embeddings.' % len(embeddings_index))

# Find Embedding vectors for our dataset and create mat
embedding_mat = numpy.zeros((top_words + index_from, embed_length))
for word, i in word_to_id.items():
    word = word.replace("'", "")

    embedding_vec = embeddings_index.get(word)
    if embedding_vec is not None and i < top_words + index_from:
        embedding_mat[i] = embedding_vec

print numpy.where(~embedding_mat.any(axis=1))[0]
# Some are zero.....

# Model Architecture
def lstm_model(neurons=8):
    model = Sequential()
    e = Embedding(top_words + index_from, embed_length, weights=[embedding_mat], input_length=max_review_length, trainable=True)
    model.add(e)
    model.add(LSTM(neurons))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=lstm_model, verbose=1)

#### Actual########
#neurons = [16]
#batch_size = [64, 128]
#epochs = [4, 6]
#folds=5

# test
neurons = [2]
batch_size = [1024]
epochs = [1]
folds=2

params = dict(neurons=neurons, epochs=epochs, batch_size=batch_size)

gridObj = GridSearchCV(estimator=model, param_grid=params, n_jobs=1, cv=folds)
result = gridObj.fit(x_train, y_train)

# final best param results
print("highest: %f using %s" % (result.best_score_, result.best_params_))

best_model = lstm_model(result.best_params_['neurons'])
best_model.fit(x_train, y_train, epochs=result.best_params_['epochs'], batch_size=result.best_params_['batch_size'])

scores = best_model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

f = open('data/CVresults.pkl', 'wb')
pkl.dump(result.best_params_, f, pkl.HIGHEST_PROTOCOL)
pkl.dump(result.cv_results_, f, pkl.HIGHEST_PROTOCOL)
f.close()

best_model.save('data/model.h5')