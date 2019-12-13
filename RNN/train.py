import torch
import torch.nn as nn
import os
from RNN_model import RNN_model
from random import random
import sys
import pickle
import argparse
from utils import preprocessing


TRANING_VALIDATION_DATA = 'traning_validation.pyc'
TESTING_DATA = 'testing.pyc'
MODEL_SAVE_PATH = 'tmp/rnn_model'

# Data & embedding configerations
PRE_TRAINED_EMBEDDING_PATH = 'glove.27B/glove.twitter.27B.200d.txt'
DATA_PATH = 'data'

def accuracy(x0, x1, y, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x0 = x0.to(device)
    x1 = x1.to(device)
    y = y.to(device)
    y_pred = model(x0, x1)
    tot = 0
    good = 0
    for i in range(len(y)):
        tot += 1
        max_prob = -1
        max_k = 0
        for j in range(6):
            if y_pred[i][j] > max_prob:
                max_prob = y_pred[i][j].item()
                max_k = j
        if (y[i].item() == max_k):
            good += 1
    return good / tot


def train(embedding_path, input_path, validation_path, dropout_rate=0, 
          batch_size=100, num_epochs=2000, learning_rate=4e-3, pretrained_model=None, 
          starting_t=0, skip_connection=False):
    try:
        with open(TRANING_VALIDATION_DATA, 'rb') as f:
            x0, x1, y, x0_val, x1_val, y_val = pickle.load(f)
    except:
        x0, x1, y = preprocessing(embedding_path, input_path)
        x0_val, x1_val, y_val = preprocessing(embedding_path, validation_path)
        with open(TRANING_VALIDATION_DATA, 'wb') as f:
            pickle.dump((x0, x1, y, x0_val, x1_val, y_val), f)
    if pretrained_model is None:
        model = RNN_model(768, 768, 6, skip_connection=skip_connection)
    else:
        model = pretrained_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x0 = x0.to(device)
    x1 = x1.to(device)
    y = y.to(device)
    max_a = 0
    # stopping_sign = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for t in range(starting_t, num_epochs):
#        permutation = torch.randperm(x0.shape[0])

 #       for i in range(0, x0.shape[0], batch_size):
        optimizer.zero_grad()

        #    indices = permutation[i:i + batch_size]
         #   if len(indices) < batch_size:
          #      continue

           # batch_x0, batch_x1, batch_y = x0[indices], x0[indices], y[indices]
        y_pred = model(x0, x1)
            # print(y_pred.size)
        loss = criterion(y_pred, y)
        print(loss)

        loss.backward()
        optimizer.step()
        a = accuracy(x0_val, x1_val, y_val, model)
        # print(loss)
        print('epoch ' + str(t) + ': training accuracy: ' + 
              str(accuracy(x0, x1, y, model)) + 
              ' | validation accuracy:  ' + 
              str(a))
        
        if t % 5 == 0:
            torch.save(model, MODEL_SAVE_PATH + str(t) + '.torch')
        if a > max_a:
            max_a = a
            torch.save(model, 'tmp/new_RNN.torch')
    # print("Maximal validation accuracy: " + str(max_a))

    return model


def sentiment_parser():
    parser = argparse.ArgumentParser(description='Train/test a review classification model')
    parser.add_argument('model_path', type=str, default='', nargs='?',
                        help='Path to the pre-trained model. If not specified, the program will start to train a new model')
    parser.add_argument('-d, --data_path', metavar='DP', type=str, default=DATA_PATH,
                        help='Path to the data directory', dest='data_path')
    parser.add_argument('-e, --embedding_path', metavar='EP', type=str, default=PRE_TRAINED_EMBEDDING_PATH,
                        help='Path to the word embedding file', dest='embedding_path')
    parser.add_argument('-n, --num_epochs', metavar='N', type=int, default=500, dest='num_epochs',
                        help='If no pre-trained model is given, train the new model in these many epochs')
    parser.add_argument('-r, --dropout_rate', metavar='DR', type=float, default=0, dest='dropout_rate',
                        help='If no pre-trained model is given, train the new model with this dropout rate')
    parser.add_argument('-l, --learning_rate', metavar='LR', type=float, default=4e-3, dest='learning_rate',
                        help='If no pre-trained model is given, train the new model with this learning_rate rate')
    parser.add_argument('-s, --starting_t', metavar='LR', type=int, default=0, dest='starting_t',
                        help='starting epoch')
    parser.add_argument('--skip_connection', action="store_true", default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = sentiment_parser()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model_path != '':
        model = torch.load(args.model_path, map_location=device)
        model = train(args.embedding_path, os.path.join(args.data_path, 'train.data'), 
                      os.path.join(args.data_path, 'dev.data'),
                      dropout_rate=args.dropout_rate, 
                      num_epochs=args.num_epochs, learning_rate=args.learning_rate, 
                      pretrained_model=model, starting_t=args.starting_t, 
                      skip_connection=args.skip_connection) 
    else:    
        model = train(args.embedding_path, os.path.join(args.data_path, 'train.data'), 
                      os.path.join(args.data_path, 'dev.data'),
                      dropout_rate=args.dropout_rate, 
                      num_epochs=args.num_epochs, learning_rate=args.learning_rate,
                      starting_t=args.starting_t, 
                      skip_connection=args.skip_connection) 
