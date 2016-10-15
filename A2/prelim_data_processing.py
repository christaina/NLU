import re
from random import shuffle
import os
import numpy as np
import data_helpers


def read_dir(path,fns,outfile):
    """
    read all files in directory and concat them in one file
    """
    print("writing all files in %s to %s"%(path,outfile))
    with open(outfile,'w') as out:
        for f in fns:
            with open(os.path.join(path,f)) as infile:
                out.write(data_helpers.clean_str(infile.read().replace("\n",'')))
            out.write("\n")

def concat_data(data_path):
    read_dir(os.path.join(test_path,'neg'),test_neg_vals,os.path.join(data_path,'test_neg.txt'))
    read_dir(os.path.join(test_path,'pos'),test_pos_vals,os.path.join(data_path,'test_pos.txt'))
    
    read_dir(os.path.join(train_path,'neg'),train_neg_vals,os.path.join(data_path,'train_neg.txt'))
    read_dir(os.path.join(train_path,'pos'),train_pos_vals,os.path.join(data_path,'train_pos.txt'))    

def write_topwords(data,outfile):
    top_words = data_helpers.get_topk_words(data)
    print("Writing top words to %s"%outfile)

    with open(outfile,'w') as o:
        for w in top_words:
            o.write("%s\n"%w)
    return top_words

def write_top_bigrams(neg_path,pos_path,outfile,cutoff=20):
    positive_examples = list(open(pos_path, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg_path, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    train_text = positive_examples+negative_examples

    top_bi = data_helpers.get_bigrams_above_cutoff(train_text,cutoff=cutoff)
    with open(outfile,'w') as o:
        for bi in top_bi:
            o.write("%s %s\n"%bi)


def write_datatopk(data,topkwords,outfile):
    print("Writing filtered data to%s"%outfile)
    print("%s lines"%len(data))
    with open(outfile,'w') as o:
        for s in data:
            curr = s
            for w in s.split(" "):
                if w not in topkwords:
                    o.write("<oov> ")
                else:
                    o.write("%s "%w)
            o.write("\n")



def save_topkwords(p='./data/'):
    positive_examples = list(open(data_path+"train_pos.txt", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(data_path+"train_neg.txt", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    train_text = positive_examples+negative_examples
    top_words = set(write_topwords(train_text,os.path.join(p,'topkwords.txt')))
    write_datatopk(positive_examples,top_words,data_path+'train_pos_top.txt')
    write_datatopk(negative_examples,top_words,data_path+'train_neg_top.txt')


def topk_testdata(p="./data/"):
    """
    filter testing data on top 10000 words
    """
    positive_examples = list(open("./data/aclImdb/test_pos.txt", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/test_neg.txt", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    write_datatopk(positive_examples,top_words,'data/aclImdb/train_pos_top.txt')
    write_datatopk(negative_examples,top_words,'data/aclImdb/train_neg_top.txt')

if __name__=='__main__':
    data_path = './data/aclImdb/'
    vocab_path = os.path.join(data_path,'imdb.vocab')
    train_path = os.path.join(data_path,'train')
    test_path = os.path.join(data_path,'test')

    train_top_pos = './data/aclImdb/train_pos_top.txt'
    train_top_neg = './data/aclImdb/train_neg_top.txt'
    try:
        train_pos_vals = [f for f in os.listdir(os.path.join(train_path,'pos'))]
        train_neg_vals = [f for f in os.listdir(os.path.join(train_path,'neg'))]
        test_pos_vals = [f for f in os.listdir(os.path.join(test_path,'pos'))]
        test_neg_vals = [f for f in os.listdir(os.path.join(test_path,'neg'))]
    except OSError:
        print "Some paths not found"
    #concat_data(data_path)
    #save_topkwords()
    #topk_testdata()
    write_top_bigrams(neg_path=train_top_neg,pos_path=train_top_pos,outfile='./data/aclImdb/bigrams.txt',cutoff=300)

