import torch
import torch.nn as nn
import os
import sys
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


# read from train/test data files and return the tuple as (label, original_sent, candsent, trendid)
def readInData(filename):

    data = []
    trends = set([])
    
    (trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = (None, None, None, None, None, None, None)
    
    for line in open(filename):
        line = line.strip()
        #read in training or dev data with labels
        if len(line.split('\t')) == 7:
            (trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = line.split('\t')
        #read in test data without labels
        elif len(line.split('\t')) == 6:
            (trendid, trendname, origsent, candsent, origsenttag, candsenttag) = line.split('\t')
        else:
            continue
        
        #if origsent == candsent:
        #    continue
        
        trends.add(trendid)
        
        if judge == None:
            data.append((judge, origsent, candsent, trendid))
            continue

        # ignoring the training/test data that has middle label 
        if judge[0] == '(':  # labelled by Amazon Mechanical Turk in format like "(2,3)"
            nYes = eval(judge)[0]  
            data.append((nYes/5.0, origsent, candsent, trendid))   
        elif judge[0].isdigit():   # labelled by expert in format like "2"
            nYes = int(judge[0])
            data.append((nYes/5.0, origsent, candsent, trendid))
                
    return data, trends

def generate_dict(embedding_path, d_model=200):
    d = {}
    embedding_list = []
    with open(embedding_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        idx = 1
        while line:
            try:
                k = line.split()
                embedding_dim = len(k[1:])
                a = [float(w) for w in k[1:]]
                if (len(a) == d_model):
                    d[k[0]] = idx
                    idx += 1
                    embedding_list.append(a)
            except:
                pass
            line = f.readline()
    tmp = []
    for i in range(d_model):
        tmp.append(0)
    embedding_list = [tmp] + embedding_list

    embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_list), padding_idx=0)

    print('Reading embedding finished.')
        
    return d, embedding


def padding(x, max_len=10000):
    # max_len = 0
    # for xx in x:
    #     if max_len < len(xx):
    #         max_len = len(xx)
    for i in range(len(x)):
        xx = x[i]
        kk = len(xx)
        x[i] = ([torch.zeros([768], dtype=torch.float)] * (max_len - kk)) + xx
    return x


def preprocessing(embedding_path, input_path, testing=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # d, embedding = generate_dict(embedding_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    x0 = []
    x1 = []
    xx0 = []
    xx1 = []
    y = []
    max_len = 0
    trends, _ = readInData(input_path)

    stop_words = set(stopwords.words('english')) 

    embedding = BertModel.from_pretrained('bert-base-uncased')
    embedding.eval()

    for trend in trends:

        s1 = '[CLS] ' + trend[1] + ' [SEP]'
        s1_indices = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s1))
        s2 = '[CLS] ' + trend[2] + ' [SEP]'
        s2_indices = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s2))

        segments_ids_1 = [0] * len(s1_indices)
        segments_ids_2 = [1] * len(s2_indices)
        encoded_layers1, _ = embedding(s1_indices, segments_ids_1)
        encoded_layers2, _ = embedding(s1_indices, segments_ids_2)

        embedding_output1 = torch.squeeze(torch.stack(encoded_layers1, dim=0), dim=1)
        embedding_output1.permute(1,0,2)
        embedding_output2 = torch.squeeze(torch.stack(encoded_layers2, dim=0), dim=1)
        embedding_output2.permute(1,0,2)

        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum1 = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in embedding_output1:

            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            
            # Use `sum_vec` to represent `token`.
            token_vecs_sum1.append(sum_vec)

        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum2 = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in embedding_output2:

            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            
            # Use `sum_vec` to represent `token`.
            token_vecs_sum2.append(sum_vec)

        x0.append(token_vecs_sum1)
        x1.append(token_vecs_sum2)

        if testing:
            y.append([0, 0])
        else:
            y.append(trend[0])

    max_len = 0
    for xx in x0 + x1:
        if max_len < len(xx):
            max_len = len(xx)
    x0 = torch.tensor(padding(x0, max_len=max_len)).to(device)
    x1 = torch.tensor(padding(x1, max_len=max_len)).to(device)

    return x0, x1, torch.tensor(y, dtype=torch.float), embedding


def wordfreq(input_path):
    trends, _ = readInData(input_path)
    stop_words = set(stopwords.words('english')) 

    word_dict = {}
    tot = 0.0

    print(len(trends))

    for trend in trends:
        for w in word_tokenize(trend[1].lower()):
            if w in stop_words:
                continue
            if w not in word_dict:
                word_dict[w] = 0
            word_dict[w] += 1
            tot += 1
        for w in word_tokenize(trend[2].lower()):
            if w in stop_words:
                continue
            if w not in word_dict:
                word_dict[w] = 0
            word_dict[w] += 1
            tot += 1

    for w in word_dict:
        word_dict[w] = word_dict[w] / tot
    # print (word_dict)

    return word_dict


if __name__ == '__main__':
    train_data = wordfreq(sys.argv[1])
    test_data = wordfreq(sys.argv[2])
    final_dict = {}
    for w in train_data:
        if w in test_data:
            final_dict[w] = (train_data[w], test_data[w])
        else:
            final_dict[w] = (train_data[w], 0.0)

    for w in test_data:
        if w in final_dict:
            continue
        final_dict[w] = (0.0, test_data[w])

    with open('fff.pkl', 'wb') as f:
        pickle.dump(final_dict, f)

    print(final_dict)