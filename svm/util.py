import pickle

PRE_TRAINED_EMBEDDING_PATH = '../glove.twitter.27B/glove.twitter.27B.25d.txt'
TRAIN_DATA_PATH = '../data/train.data'
TEST_DATA_PATH = '../data/test.data'

class Data_processing:
    def __init__(self, embedding_path):
        self.vectors = []
        self.word2idx = {}
        self.embedding_path = embedding_path

    def create_embedding(self):
        idx = 0
        self.word2idx = {}

        with open(self.embedding_path, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                self.word2idx[word] = idx
                idx += 1
                vect = [float(w) for w in line[1:]]
                self.vectors.append(vect)

        UNKA = []
        for i in range(50):
            UNKA.append(float(0))
        self.vectors = [UNKA] + self.vectors
        return self.vectors, self.word2idx

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
            if nYes >= 3:
                amt_label = True
                data.append((amt_label, origsent, candsent, trendid))
            elif nYes <= 1:
                amt_label = False
                data.append((amt_label, origsent, candsent, trendid))
        elif judge[0].isdigit():   # labelled by expert in format like "2"
            nYes = int(judge[0])
            if nYes >= 4:
                expert_label = True
                data.append((expert_label, origsent, candsent, trendid))
            elif nYes <= 2:
                expert_label = False
                data.append((expert_label, origsent, candsent, trendid))
            else:
                expert_label = None
                data.append((expert_label, origsent, candsent, trendid))

    return data, trends

if __name__ == "__main__":
    # data_processor = Data_processing(PRE_TRAINED_EMBEDDING_PATH)
    # vectors, word2idx = None, None
    # try:
    #     file_vectors = open("vectors.pkl", "rb")
    #     file_word2idx = open("word2idx.pkl", "rb")
    #     vectors = pickle.load(file_vectors)
    #     word2idx = pickle.load(file_word2idx)
    # except:
    #     vectors, word2idx = data_processor.create_embedding()
    #     file_vectors = open("vectors.pkl", "wb+")
    #     file_word2idx = open("word2idx.pkl", "wb+")
    #     pickle.dump(vectors, file_vectors)
    #     pickle.dump(word2idx, file_word2idx)
    data, trends = readInData(TRAIN_DATA_PATH)
    print(data[0:20])
    # print(trends)
