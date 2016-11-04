import tensorflow as tf
import time
import reader
import os
import numpy as np
from scipy import spatial
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import pylab as plt

flags = tf.flags

flags.DEFINE_string('checkpoint_path',None,'Directory with checkpoint of model to load')
flags.DEFINE_string('data_path','../simple-examples/data','Directory with checkpoint of model to load')

FLAGS = flags.FLAGS


def word_lookup(word,train_path=os.path.join(FLAGS.data_path,'ptb.train.txt')):
    vocab = reader._build_vocab(train_path)
    if word in vocab:
        return vocab[word]
    else:
        print("word not in vocab.")
        return None

def similarity(model,w1,w2):
     emb = tf.get_default_graph().get_tensor_by_name("Model/embedding:0")
     tensored = tf.convert_to_tensor([word_lookup(w1),word_lookup(w2)],name='embeddings')
     input = tf.nn.embedding_lookup(emb,tensored).eval()
     #dist = 1-spatial.distance.cosine(input[0],input[1])
     dist = cosine_similarity(input[0],input[1])
     return dist


def score(model):
    score = 0.
    score += similarity(model,'a', 'an') > similarity(model,'a', 'document')
    score += similarity(model,'in', 'of') > similarity(model,'in', 'picture')
    score += similarity(model,'nation', 'country') > similarity(model,'nation', 'end')
    score += similarity(model,'films', 'movies') > similarity(model,'films', 'almost')
    score += similarity(model,'workers', 'employees') > similarity(model,'workers', 'movies')
    score += similarity(model,'institutions', 'organizations') > similarity(model,'institution', 'big')
    score += similarity(model,'assets', 'portfolio') > similarity(model,'assets', 'down')
    #score += similarity(model,"'", ",") > similarity(model,"'", 'quite')
    score += similarity(model,'finance', 'acquisition') > similarity(model,'finance', 'seems')
    score += similarity(model,'good', 'great') > similarity(model,'good', 'minutes')
    return score

def run_TSNE(model):
    print("Running TSNE on word embeddings")
    emb = tf.get_default_graph().get_tensor_by_name("Model/embedding:0").eval()
    tsne = TSNE(learning_rate=300)
    st = time.time()
    transformed = tsne.fit_transform(emb)
    print("Finished in %s"%(time.time()-st))
    return transformed.T

def viz_TSNE(tsne_emb,save_path,train_path=os.path.join(FLAGS.data_path,'ptb.train.txt'),samples=500):
    """
    Makes visualization of random sample of t-SNE embedding and annotate with words, saves.
    ------
    parameters

    tsne_emb: array of t-SNE embedding
    save_path: path to save visualization to
    train path: path to training data
    samples: number of embeddings to sample for visualization
    """

    vocab = reader._build_vocab(train_path)
    voc_reverse = {v:k for k,v in vocab.iteritems()}
    r_dx = np.random.choice(tsne_emb.T.shape[0], samples)
    embs_rand = tsne_emb.T[r_dx].T
    keys = np.array(voc_reverse.keys())[r_dx]
    plt.figure(figsize=(20,20))

    plt.scatter(embs_rand[0],embs_rand[1])
    plt.title("Word embeddings from random %s words using t-SNE"%samples)

    for i, txt in enumerate(embs_rand.T):
        plt.annotate(voc_reverse[keys[i]], (embs_rand[0][i],embs_rand[1][i]))
        plt.savefig(save_path)

def main(_):
    if not FLAGS.checkpoint_path:
        print("Missing checkpoint path")

    else:
        print("Loading from %s"%FLAGS.checkpoint_path)
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            model = tf.get_default_graph()
            print('Final Score: %s out of 9'%score(model))
            tsne_data = run_TSNE(model)
            np.savetxt(os.path.join(FLAGS.checkpoint_path,'tsne_proj.txt'),tsne_data)
            viz_TSNE(tsne_data,os.path.join(FLAGS.checkpoint_path,'tsne.png'),samples=500)


if __name__=="__main__":
    tf.app.run()
