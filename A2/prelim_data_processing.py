import re
from random import shuffle
import os
import numpy as np
import data_helpers

data_path = './data/aclImdb/'
vocab_path = os.path.join(data_path,'imdb.vocab')
train_path = os.path.join(data_path,'train')
test_path = os.path.join(data_path,'test')
train_pos_vals = [f for f in os.listdir(os.path.join(train_path,'pos'))]
train_neg_vals = [f for f in os.listdir(os.path.join(train_path,'neg'))]
test_pos_vals = [f for f in os.listdir(os.path.join(test_path,'pos'))]
test_neg_vals = [f for f in os.listdir(os.path.join(test_path,'neg'))]

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

def write_list(li,outfile):
    with open(outfile,'w') as o:
        for w in li:
            o.write("%s\n"%w)

def save_topkwordsbi(p='./data/'):
    train_text,y = data_helpers.load_data_and_labels()
    top_words = data_helpers.get_topk_words(train_text)
    write_list(top_words,os.path.join(p,'topkwords.txt'))
    top_bi = data_helpers.get_topk_bigrams(train_text)
    write_list(top_bi,os.path.join(p,'topkbi.txt'))


 if __name__=='__main__':
    save_topkwordsbi()
    #concat_data(data_path)

