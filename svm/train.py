from util import *
from ngram import *
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

USE_EMBEDDING = True

def get_train_X_Y(filename, vectors, word2idx, C, V):
    data, _ = readInData(filename)
    X = []
    Y = []
    scaler = StandardScaler()
    for data_row in data:
        sent_1 = data_row[1].lower()
        sent_2 = data_row[2].lower()
        label = data_row[0]
        if vectors != None and word2idx != None:
            X.append(get_ngram_features(gen_chars_ngrams, sent_1, sent_2, C) + \
                     get_ngram_features(gen_word_ngrams, sent_1, sent_2, V) + \
                     word2vec_features(vectors, word2idx, sent_1, sent_2))
        else:
            X.append(get_ngram_features(gen_chars_ngrams, sent_1, sent_2, C) + \
                     get_ngram_features(gen_word_ngrams, sent_1, sent_2, V))
        Y.append(label)
    X = scaler.fit_transform(np.asarray(X))
    Y = np.asarray(Y)
    return X, Y, scaler

def get_test_X_Y(filename, scaler, vectors, word2idx, C, V):
    data, _ = readInData(filename)
    X = []
    Y = []
    for data_row in data:
        sent_1 = data_row[1].lower()
        sent_2 = data_row[2].lower()
        label = data_row[0]
        if vectors != None and word2idx != None:
            X.append(get_ngram_features(gen_chars_ngrams, sent_1, sent_2, C) + \
                     get_ngram_features(gen_word_ngrams, sent_1, sent_2, V) + \
                     word2vec_features(vectors, word2idx, sent_1, sent_2))
        else:
            X.append(get_ngram_features(gen_chars_ngrams, sent_1, sent_2, C) + \
                     get_ngram_features(gen_word_ngrams, sent_1, sent_2, V))
        Y.append(label)
    X = scaler.transform(np.asarray(X))
    Y = np.asarray(Y)
    return X, Y

def norm(v):
    return math.sqrt(np.dot(v, v))

def get_words_vec(vectors, word2idx, sentence):
    return np.sum([vectors[word2idx[word]] for word in sentence.split() if word in word2idx], axis=0)

def word2vec_features(vectors, word2idx, sentence_1, sentence_2):
    u = get_words_vec(vectors, word2idx, sentence_1)
    v = get_words_vec(vectors, word2idx, sentence_2)
    c = norm(u)
    d = norm(v)
    if c == 0 or d == 0:
        print(sentence_1, sentence_2)
        return [0]
    else:
        return [np.dot(u, v) / c / d]

def eval(result, truth):
    tp, tn, fp, fn = [0, 0, 0, 0]
    correct = 0
    total = len(result)
    for i in range(len(result)):
        guess, answer = result[i], truth[i]
        if guess == True and answer == False:
            fp += 1.0
        elif guess == False and answer == True:
            fn += 1.0
        elif guess == True and answer == True:
            tp += 1.0
            correct += 1
        elif guess == False and answer == False:
            tn += 1.0
            correct += 1
    accuracy = correct / total
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn)  if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1

def train(vectors, word2idx, C, V):
    train_X, train_Y, scaler = get_train_X_Y(TRAIN_DATA_PATH, vectors, word2idx, C, V)
    clf = AdaBoostClassifier(n_estimators=256)
    # clf = LinearSVC(max_iter=10000, C = 2.0)
    # clf = SVC(C = 2.0, kernel='rbf')
    clf.fit(train_X, train_Y)
    return clf, scaler, eval(clf.predict(train_X), train_Y)

def test(vectors, word2idx, C, V):
    clf, scaler, train_eval = train(vectors, word2idx, C, V)
    print("training accuracy is: ", round(train_eval[0], 5))
    print("training f1 is: ", round(train_eval[-1], 5))
    test_X, test_Y = get_test_X_Y(TEST_DATA_PATH, scaler, vectors, word2idx, C, V)
    return eval(clf.predict(test_X), test_Y)

def main():
    data_processor = Data_processing(PRE_TRAINED_EMBEDDING_PATH)
    vectors, word2idx = None, None

    if USE_EMBEDDING:
        try:
            file_vectors = open("vectors.pkl", "rb")
            file_word2idx = open("word2idx.pkl", "rb")
            vectors = pickle.load(file_vectors)
            word2idx = pickle.load(file_word2idx)
        except:
            vectors, word2idx = data_processor.create_embedding()
            file_vectors = open("vectors.pkl", "wb+")
            file_word2idx = open("word2idx.pkl", "wb+")
            pickle.dump(vectors, file_vectors)
            pickle.dump(word2idx, file_word2idx)
            file_vectors = open("vectors.pkl", "rb")
            file_word2idx = open("word2idx.pkl", "rb")
            vectors = pickle.load(file_vectors)
            word2idx = pickle.load(file_word2idx)
        print("file reading completed")

    Cs = [1, 2]
    Vs = [1, 2]
    for C in Cs:
        for V in Vs:
            print("C = " + str(C), "V = " + str(V))
            print(test(vectors, word2idx, C, V))
            print("================================")

if __name__ == "__main__":
    main()
