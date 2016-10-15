import numpy as np
from itertools import izip, islice
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def split_data(data,labels,val_pct=0.2):
    shuffle = np.random.seed(10)
    shuffle_ind = np.random.permutation(np.arange(len(labels)))
    data_shuf = data[shuffle_ind]
    label_shuf = labels[shuffle_ind]
    
    val_n = int(len(labels)*val_pct)
    
    val_data,val_labels = data_shuf[:val_n],label_shuf[:val_n]
    train_data,train_labels = data_shuf[val_n:],label_shuf[val_n:]
    return (val_data,val_labels),(train_data,train_labels)

def get_bigram_cutoff(data,k=2000):
    words = (" ").join(data).split(" ")
    c = Counter(izip(words, islice(words, 1, None)))
    common_bi = c.most_common(k)
    return common_bi[-1][1]

def get_bigrams_above_cutoff(data,cutoff):
    words = (" ").join(data).split(" ")
    c = Counter(izip(words, islice(words, 1, None)))
    # get bigrams above cutoff
    cutoff_bigrams = [k for k,v in c.iteritems() if v>cutoff]
    return cutoff_bigrams

def load_bigram_feats(posfn,negfn):
    # assumes everything is in same order as data
    pos_bigrams = load_bigrams_file(posfn)
    neg_bigrams = load_bigrams_file(negfn)
    all_bigrams = pos_bigrams+neg_bigrams
    return all_bigrams

def load_bigrams_file(path):
    bigrams=[]
    with open(path,'r') as o:
        for l in o.readlines():
            l = tuple(l.strip("\n").split(" "))
            bigrams.append(l)
    return bigrams

    
def bigram_feats(data,bigrams_list):
    """
    bigrams_list: list with the bigrams we want to use
    """
    row_split = [r.split(" ") for r in data]
    m = [zip(r, r[1:]) for r in row_split]
    #words = (" ").join(data).split(" ")
    #c = Counter(izip(words, islice(words, 1, None)))
    #x_bi = [zip(s, s[1:]) for s in words]
    r_bi = [[bi for bi in r if bi in bigrams_list] for r in m]
    #for r in m:
        #print(r)
    #    r_bi = [bi for bi in r if bi in bigrams_list]
    #    bigrammed.append(r_bi)
    return r_bi

def get_topk_words(data,k=10000):
    data_concat = (" ").join(data).split(" ")
    word_counts = Counter(data_concat)
    common_words = ([c[0] for c in word_counts.most_common(k)])
    return common_words

def load_bigrams_file(path):
    bigrams = []
    with open(path) as o:
        for line in o.readlines():
            line = line.strip("\n").split("$$$")
            bigrams.append(line)
    return bigrams


def get_topk_bigrams(data,k=2000):
    words = (" ").join(data).split(" ")
    c = Counter(izip(words, islice(words, 1, None)))
    common_bi = [c[0] for c in c.most_common(k)]
    return common_bi

def add_ood(data, topkwords):
    data_top = []
    for s in data:
        curr = s
        for w in s.split(" "):
            if w not in topkwords:
                curr = curr.replace(w,"<oov>")
        data_top.append(curr)
    return data_top


def load_data_and_labels(posfn,negfn):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(negfn, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(posfn, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
