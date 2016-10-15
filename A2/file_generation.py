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

def concat_set(dir,outfile):
    print("writing all files in %s to %s"%(dir,outfile))
    fns = [os.path.join(dir,f) for f in os.listdir(dir)]
    with open(outfile,'w') as o:
        for f in fns:
            with open(f) as infile:
                o.write(data_helpers.clean_str(infile.read().replace("\n",'')))
            o.write("\n")

def write_top_words(neg_path,pos_path,outfile,k=10000):
    all_text = data_helpers.load_data_and_labels(neg_path,pos_path)[0]
    top_words = data_helpers.get_topk_words(all_text,k=k)
    print("Writing top words to %s"%outfile)

    with open(outfile,'w') as o:
        for w in top_words:
            o.write("%s\n"%w)
    return top_words

def write_top_bigrams(neg_path,pos_path,outfile,cutoff=200):
    all_text = data_helpers.load_data_and_labels(neg_path,pos_path)[0]
    print("Writing bigrams with count above %s to %s"%(cutoff,outfile))
    top_bi = data_helpers.get_bigrams_above_cutoff(all_text,cutoff=cutoff)
    print("Result: %s bigrams"%len(top_bi))
    with open(outfile,'w') as o:
        for bi in top_bi:
            o.write("%s %s\n"%bi)

def write_datatopk(path, topkpath,outfile):
    data = list(open(path, "r").readlines())
    data = [s.strip() for s in data]
    print("Writing filtered data to%s - %s lines"%(outfile,len(data)))
    topwords = list(open(topkpath,'r').readlines())
    topwords = [s.strip() for s in topwords]
    with open(outfile,'w') as o:
        for s in data:
            curr = s
            for w in s.split(" "):
                if w not in topwords:
                    o.write("<oov> ")
                else:
                    o.write("%s "%w)
            o.write("\n")

def write_data_bigrams(path, topbipath,outfile):
    data = list(open(path, "r").readlines())
    data = [s.strip() for s in data]
    print("Writing filtered bigram features to%s - %s lines"%(outfile,len(data)))
    # load top bigrams
    topbi = list(open(topbipath,'r').readlines())
    # turn bigram lines into tuples
    topbi = [tuple(s.strip().split(" ")) for s in topbi]
    print("Found %s bigrams to filter on!"%len(topbi))

    #split data by row
    data_split =[r.split(" ") for r in data]

    # get bigrams from each row
    bigram_data = [zip(r,r[1:]) for r in data_split]
    # filter on top bigrams
    filtered_bigram_data = [[bi for bi in r if bi in topbi] for r in bigram_data]

    with open(outfile,'w') as o:
        # for row
        for r in filtered_bigram_data:
            # for each bigram
            for t in r:
                # to accurately store bigrams
                o.write("%s %s$$$"%t)
            o.write("\n")

def save_topkwords(p='./data/'):
    positive_examples = list(open(data_path+"train_pos.txt", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(data_path+"train_neg.txt", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    train_text = positive_examples+negative_examples
    top_words = write_topwords(train_text,os.path.join(p,'topkwords.txt'))
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

def pa(data_path,fn):
    return os.path.join(data_path,fn)

if __name__=='__main__':
    datap = './data/aclImdb/'

    # origin data paths
    tr_pos_dir = pa(datap,'train/pos/')
    tr_neg_dir = pa(datap,'train/neg/')
    te_pos_dir = pa(datap,'test/pos/')
    te_neg_dir = pa(datap,'test/neg/')
    
    # concatenating all the separate files
    train_pos = pa(datap,'train_pos.txt')
    train_neg = pa(datap,'train_neg.txt')
    test_pos = pa(datap,'test_pos.txt')
    test_neg = pa(datap,'test_neg.txt')
    
    # files containing bigrams and words we want to use
    topkwords = pa(datap,'topk.txt')
    bigrams = pa(datap,'bigrams.txt')

    # data files filtered on top k words
    tr_pos_top = pa(datap,'train_pos_top.txt')
    tr_neg_top = pa(datap,'train_neg_top.txt')
    te_pos_top = pa(datap,'te_pos_top.txt')
    te_neg_top = pa(datap,'te_neg_top.txt')

    # files with bigrams
    tr_pos_bi = pa(datap,'train_pos_bi.txt')
    tr_neg_bi = pa(datap,'train_neg_bi.txt')
    te_pos_bi = pa(datap,'te_pos_bi.txt')
    te_neg_bi = pa(datap,'te_neg_bi.txt')

    concat_set(tr_pos_dir,train_pos)
    concat_set(tr_neg_dir,train_neg)
    concat_set(te_pos_dir,test_pos)
    concat_set(te_neg_dir,test_neg)

    write_top_words(pos_path=train_pos,neg_path=train_neg,outfile=topkwords)
    write_top_bigrams(pos_path=train_pos,neg_path=train_neg,outfile=bigrams,cutoff=300)

    write_datatopk(train_pos,topkwords,outfile=tr_pos_top)
    write_datatopk(train_neg,topkwords,outfile=tr_neg_top)
    write_datatopk(test_pos,topkwords,outfile=te_pos_top)
    write_datatopk(test_neg,topkwords,outfile=te_neg_top)

    write_data_bigrams(tr_pos_top,bigrams,outfile=tr_pos_bi)
    write_data_bigrams(tr_neg_top,bigrams,outfile=tr_neg_bi)
    write_data_bigrams(te_pos_top,bigrams,outfile=te_pos_bi)
    write_data_bigrams(te_neg_top,bigrams,outfile=te_neg_bi)

    #concat_data(data_path)
    #save_topkwords()
    #topk_testdata()
    #write_top_bigrams(neg_path=train_top_neg,pos_path=train_top_pos,outfile='./data/aclImdb/bigrams.txt',cutoff=300)

