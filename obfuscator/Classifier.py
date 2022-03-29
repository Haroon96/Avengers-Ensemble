from asyncore import write
import pickle
from writeprintsStatic import writeprintsStatic
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import operator
import io
import os
from pickle import load
import numpy as np
from keras.models import model_from_json
import pickle

def load_dataset(filename):
    return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def fill_in_missing_words_with_zeros(embeddings_index, word_index, EMBEDDING_DIM):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])

def load_model(filename):
    embeddings_index = {}
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            if len(coefs) == 300:
                embeddings_index[word] = coefs
        except:
            # print(values)
            c = 1
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index



# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

def getData(datasetName, authorsRequired):
    
    picklesPath = '../datasets/' + datasetName + "-" + str(authorsRequired) + "/"
    with open(picklesPath+'X_train.pickle', 'rb') as handle:
        X_train = pickle.load(handle)

    with open(picklesPath+'X_test.pickle', 'rb') as handle:
        X_test = pickle.load(handle)
    
    
    return (X_train, X_test)
  
def getAllData(datasetName, authorsRequired):
    (X_train_all, X_test_all) = getData(datasetName, authorsRequired)
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for (filePath, filename, authorId, author, inputText) in X_train_all:
        X_train.append(inputText)
        y_train.append(authorId)
    for (filePath, filename, authorId, author, inputText) in X_test_all:
        X_test.append(inputText)
        y_test.append(authorId)

    return np.matrix(X_train), np.matrix(X_test), y_train, y_test
class Classifier:

    def __init__(self, classifierType, authorstoKeep, datasetName, indexNumber):

        self.classifierType = classifierType
        self.authorstoKeep = authorstoKeep
        self.datasetName = datasetName
        self.clf = None
        self.tokenizer = None
        self.MAX_SEQUENCE_LENGTH = None
        self.indexNumber = indexNumber

    def loadClassifier(self):

        if self.classifierType == 'ml':
            classifierName = "../classifier/trainedModels/" + str(self.datasetName) + '-' + str(self.authorstoKeep) + f'/trained_model.sav'
            self.clf = pickle.load(open(classifierName, 'rb'))

    def getLabelAndProbabilities(self, document):

        if self.classifierType == 'ml':
            tempFile = open(str(self.indexNumber) + "tempText", "w")
            tempFile.write(document.documentText)
            tempFile.close()

            inputText = io.open(str(self.indexNumber) + "tempText" ,"r", errors="ignore").readlines()
            inputText = ''.join(str(e) + "" for e in inputText)

            # get writeprints features
            features = np.asarray([writeprintsStatic.calculateFeatures(inputText)])

            # get prediction probabilities
            prediction_probability = list(self.clf.predict_proba(features)[0])

            # get author probabilities and final prediction
            document.documentAuthorProbabilites = {key: value for (key, value) in enumerate(prediction_probability )}
            document.documentAuthor = self.clf.predict(features)[0]

            os.remove(str(self.indexNumber) + "tempText")

        elif self.classifierType == 'dl':
            sequences = self.tokenizer.texts_to_sequences([document.documentText])
            data = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)

            document.documentAuthorProbabilites = {key: value for (key, value) in enumerate(
                list(self.clf.predict_proba([np.array(data), np.array(data)])[0]))}

            document.documentAuthor, _ = max(document.documentAuthorProbabilites.items(), key=lambda x: x[1])
