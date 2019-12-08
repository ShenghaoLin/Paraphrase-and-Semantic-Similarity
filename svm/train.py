from util import *
from ngram import *
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

USE_EMBEDDING = False

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

def train(train_X, train_Y, clf, C, V):
    clf.fit(train_X, train_Y)
    train_eval = eval(clf.predict(train_X), train_Y)
    print("training accuracy is: ", round(train_eval[0], 5))
    print("training f1 is: ", round(train_eval[-1], 5))
    return clf

def test(test_X, test_Y, clf, C, V):
    pred_Y = clf.predict(test_X)
    return pred_Y, eval(pred_Y, test_Y)

def main(use_embedding=USE_EMBEDDING):
    data_processor = Data_processing(PRE_TRAINED_EMBEDDING_PATH)
    vectors, word2idx = None, None

    if use_embedding:
        try:
            file_vectors = open(PKL_PATH + "vectors.pkl", "rb")
            file_word2idx = open(PKL_PATH + "word2idx.pkl", "rb")
            vectors = pickle.load(file_vectors)
            word2idx = pickle.load(file_word2idx)
        except:
            vectors, word2idx = data_processor.create_embedding()
            file_vectors = open(PKL_PATH + "vectors.pkl", "wb+")
            file_word2idx = open(PKL_PATH + "word2idx.pkl", "wb+")
            pickle.dump(vectors, file_vectors)
            pickle.dump(word2idx, file_word2idx)
            file_vectors.close()
            file_word2idx.close()
            file_vectors = open(PKL_PATH + "vectors.pkl", "rb")
            file_word2idx = open(PKL_PATH + "word2idx.pkl", "rb")
            vectors = pickle.load(file_vectors)
            word2idx = pickle.load(file_word2idx)
            file_vectors.close()
            file_word2idx.close()
        # print("file reading completed")

    pipes = {'ada': {}, 'LinearSVC': {}, 'rbf': {}}
    # AdaBoostClassifier
    pipes['ada']['pipeline'] = Pipeline([
        ("clf", AdaBoostClassifier())
    ])

    pipes['ada']['param_grid'] = {
        'clf__n_estimators':[56, 128, 256]
    }
    # LinearSVC
    pipes['LinearSVC']['pipeline'] = Pipeline([
        ("clf", LinearSVC())
    ])

    pipes['LinearSVC']['param_grid'] = {
        'clf__dual':[False],
        'clf__C':[0.2, 0.5, 1.0, 2.0],
        'clf__penalty':["l2", "l1"],
        'clf__max_iter':[10000]
    }
    # rbf
    pipes['rbf']['pipeline'] = Pipeline([
        ("clf", SVC())
    ])

    pipes['rbf']['param_grid'] = {
        'clf__kernel':["rbf"],
        'clf__C':[0.5, 1.0, 2.0],
        'clf__gamma':['auto']
    }

    Cs = [1, 2]
    Vs = [1, 2]
    for clf_name in pipes:
        for C in Cs:
            for V in Vs:
                print("================" + clf_name + "_C" + str(C) + "V" + str(V) + "================")
                train_X, train_Y, scaler = get_train_X_Y(TRAIN_DATA_PATH, vectors, word2idx, C, V)
                test_X, test_Y = get_test_X_Y(TEST_DATA_PATH, scaler, vectors, word2idx, C, V)
                pipeline = pipes[clf_name]['pipeline']
                param_grid = pipes[clf_name]['param_grid']

                grid = GridSearchCV(pipeline, cv=5, param_grid=param_grid, scoring="f1", return_train_score=True)
                grid.fit(train_X, train_Y)
                print("------------training------------")
                print("Best params: %s" % (grid.best_params_))
                # print(grid.cv_results_)
                print("Validation score:", grid.best_score_)
                clf = grid.best_estimator_
                train_result_Y = clf.predict(train_X)
                train_eval = eval(train_result_Y, train_Y)
                print("training result:", train_eval)
                print("------------testing------------")
                pred_Y, eval_result = test(test_X, test_Y, clf, C, V)
                print("testing result:", eval_result)
                print("test score:", eval_result[-1])
                print()

                output = open("../output/trainditional_methods_with_embedding_25d/PIT2015_03_" + clf_name + "_C" + str(C) + "V" + str(V) + ".output", "w+")
                for Y in pred_Y:
                    if Y:
                        output.write('true\t')
                    else:
                        output.write('false\t')
                    output.write('0.0000\n')

if __name__ == "__main__":
    main(True)
