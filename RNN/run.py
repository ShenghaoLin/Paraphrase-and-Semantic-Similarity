import argparse
import torch
import pickle
from utils import preprocessing


TESTING_DATA = 'testing.pyc'
DATA_PATH = 'data/test.data'
OUTPUT_PATH = 'output/PIT2015_02_rnn.output'
PRE_TRAINED_EMBEDDING_PATH = 'glove.27B/glove.twitter.27B.200d.txt'


def run_model(embedding_path, input_path, output_path, model):
    try:
        with open(TESTING_DATA, 'rb') as f:
            x0, x1 = pickle.load(f)
    except:
        x0, x1, _ = preprocessing(embedding_path, input_path, testing=True)
        with open(TESTING_DATA, 'wb') as f:
            pickle.dump((x0, x1), f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x0 = x0.to(device)
    x1 = x1.to(device)
    y_pred = model(x0, x1)
    
    with open(output_path, 'w') as f:
        for y in y_pred:
            max_prob = -1
            max_k = -1
            for j in range(6):
                if y[j] > max_prob:
                    max_prob = y[j].item()
                    max_k = j
            if max_k < 3:
                f.write('true\t')
            else:
                f.write('false\t')
            f.write(max_k / 5.0 + '\n')


def sentiment_parser():
    parser = argparse.ArgumentParser(description='Test a review classification model')
    parser.add_argument('model_path', type=str, default='tmp/model.torch', nargs='?',
                        help='Path to the pre-trained model')
    parser.add_argument('-d, --data_path', metavar='DP', type=str, default=DATA_PATH,
                        help='Path to the data', dest='data_path')
    parser.add_argument('-p, --output_path', metavar='OP', type=str, default=OUTPUT_PATH,
                        help='Path to the output', dest='output_path')
    parser.add_argument('-e, --embedding_path', metavar='EP', type=str, default=PRE_TRAINED_EMBEDDING_PATH,
                        help='Path to the word embedding file', dest='embedding_path')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = sentiment_parser()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path, map_location=device)
    run_model(args.embedding_path, args.data_path, args.output_path, model)
