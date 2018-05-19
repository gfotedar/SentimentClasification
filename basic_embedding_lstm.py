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

# fix seed
numpy.random.seed(1)

# Using only top 10k words instead of all
top_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words, skip_top=20)

#Pad smaller sequences
max_review_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

#Word Embeddings vector length
embed_length = 100

#Model Architecture
def lstm_model(neurons=8):
    model = Sequential()
    model.add(Embedding(top_words, embed_length, input_length=max_review_length))
    model.add(LSTM(neurons))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=lstm_model, verbose=1)

#Hyperparameter values to search among
#neurons = [8, 16, 64, 128]
#batch_size = [64, 128, 256, 512]
#epochs = [4, 6, 8]


#### Actual########
#neurons = [16]
#batch_size = [64, 128]
#epochs = [4, 6]
#folds=5

# test
neurons = [2]
batch_size = [1024]
epochs = [1]
folds = 2

params = dict(neurons=neurons, epochs=epochs, batch_size=batch_size)

#Crossvalidation Search for hyperparams
gridObj = GridSearchCV(estimator=model, param_grid=params, n_jobs=1, cv=folds)
result = gridObj.fit(x_train, y_train)

# final best param results
print("Highest: %f using %s" % (result.best_score_, result.best_params_))

best_model=lstm_model(result.best_params_['neurons'])
best_model.fit(x_train, y_train, epochs=result.best_params_['epochs'], batch_size=result.best_params_['batch_size'])

scores = best_model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

f = open('data/CVresults.pkl', 'wb')
pkl.dump(result.best_params_,f,pkl.HIGHEST_PROTOCOL)
pkl.dump(result.cv_results_,f,pkl.HIGHEST_PROTOCOL)
f.close()

best_model.save('data/model.h5')


#model = load_model('data/model.h5')
#scores = model.evaluate(X_test, y_test, verbose=1)
#print("Accuracy: %.2f%%" % (scores[1]*100))

#f = open('data/CVresults.pkl', 'rb')
#for i in range(2):
#    print pkl.load(f)

#f.close()