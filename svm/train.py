from util import *
from ngram import *
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

def get_train_X_Y(filename, file_vectors, file_word2idx):
    data, _ = readInData(filename)
    X = []
    Y = []
    scaler = StandardScaler()
    for data_row in data:
        sent_1 = data_row[1]
        sent_2 = data_row[2]
        label = data_row[0]
        X.append(get_ngram_features(gen_chars_ngrams, sent_1, sent_2, 2) + \
                 get_ngram_features(gen_word_ngrams, sent_1, sent_2, 1)) + \
                 word2vec_features(file_vectors, file_word2idx, sent_1, sent_2)
        Y.append(label)
    X = scaler.fit_transform(np.asarray(X))
    Y = np.asarray(Y)
    return X, Y, scaler

def get_test_X_Y(filename, scaler, file_vectors, file_word2idx):
    data, _ = readInData(filename)
    X = []
    Y = []
    for data_row in data:
        sent_1 = data_row[1]
        sent_2 = data_row[2]
        label = data_row[0]
        X.append(get_ngram_features(gen_chars_ngrams, sent_1, sent_2, 2) + \
                 get_ngram_features(gen_word_ngrams, sent_1, sent_2, 1)) + \
                 word2vec_features(file_vectors, file_word2idx, sent_1, sent_2)
        Y.append(label)
    X = scaler.transform(np.asarray(X))
    Y = np.asarray(Y)
    return X, Y

def norm(v):
    return math.sqrt(np.dot(v, v))

def get_words_vec(file_vectors, file_word2idx, sentence):
    vectors = pickle.load(file_vectors)
    word2idx = pickle.load(file_word2idx)
    return sum(vectors[word2idx[word]] for word in sentence.split() if word in word2idx)

def word2vec_features(file_vectors, file_word2idx, sentence_1, sentence_2):
    u = get_words_vec(file_vectors, file_word2idx, sentence_1)
    v = get_words_vec(file_vectors, file_word2idx, sentence_2)
    c = norm(u)
    d = norm(v)
    if c == 0 or d == 0:
        print("warning: got zero normal")
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

def train(file_vectors, file_word2idx):
    train_X, train_Y, scaler = get_train_X_Y(TRAIN_DATA_PATH, file_vectors, file_word2idx)
    # clf = AdaBoostClassifier(n_estimators=50)
    # clf = LinearSVC(max_iter=10000)
    clf = SVC(C=1.0, kernel='rbf')
    clf.fit(train_X, train_Y)
    return clf, scaler, eval(clf.predict(train_X), train_Y)

def test(file_vectors, file_word2idx):
    clf, scaler, train_eval = train()
    print("training accuracy is: ", round(train_eval[0], 5))
    print("training f1 is: ", round(train_eval[-1], 5))
    test_X, test_Y = get_test_X_Y(TEST_DATA_PATH, scaler, file_vectors, file_word2idx)
    return eval(clf.predict(test_X), test_Y)


if __name__ == "__main__":
    data_processor = Data_processing(PRE_TRAINED_EMBEDDING_PATH)
    vectors, word2idx = None, None
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
    print(test())
