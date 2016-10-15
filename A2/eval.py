#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("cutoff", 50, "When to cut off sentence")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/nobi_c50_e30/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("bigrams", False, "Use bigram features or not")

test_pos = './data/aclImdb/te_pos_top.txt'
test_neg = './data/aclImdb/te_neg_top.txt'
test_bigram_pos = './data/aclImdb/te_pos_bi.txt'
test_bigram_neg = './data/aclImdb/te_neg_bi.txt'

bigrams_file = './data/aclImdb/bigrams.txt'

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

x_raw,y_test = data_helpers.load_data_and_labels(test_pos,test_neg)
x_raw = [(" ").join(x.split(" ")[:FLAGS.cutoff]) for x in x_raw]
y_test = [y[1] for y in y_test]


#print("\nParameters:")
#for attr, value in sorted(FLAGS.__flags.items()):
#    print("{}={}".format(attr.upper(), value))
#print("")

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "../vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

vocab_size = len(vocab_processor.vocabulary_)

if (FLAGS.bigrams==True):
    bigrams_features = data_helpers.load_bigram_feats(test_bigram_pos,test_bigram_neg)
    bigrams_list = (data_helpers.load_bigrams_file(bigrams_file))
    bigrams_list = [" ".join(i) for i in bigrams_list]
    #bigrams_list = set(bigrams_list)
    vocab_size = vocab_size+len(bigrams_list)+1
    bigram_st = len(vocab_processor.vocabulary_)+1
    max_bigram_length = max([len(bi) for bi in bigrams_features])
    bigrams_mat = np.empty([len(bigrams_features),max_bigram_length])

    bigrams_dict = {}

    for i,r in enumerate(bigrams_features):
        for j,w in enumerate(r):
            if w == "":
                bigrams_mat[i][j]=0
            else:

                bigrams_mat[i][j]=bigrams_list.index(w)+bigram_st
    x_test = np.concatenate((x_test,bigrams_mat),axis=1)

#print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        shape_cutoff = (input_x.get_shape())[1]
        if shape_cutoff < len(x_test[0]):
            x_test = [x[:shape_cutoff] for x in x_test]
        #input_y = graph.get_operation_by_name("input_y").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:

    correct_predictions = float(sum([pred==true for pred,true in zip(all_predictions,y_test)]))
    print("")
    #print("Total number of test examples: {}".format(len(y_test)))
    print("Model: {}, Accuracy: {:g}".format(FLAGS.checkpoint_dir,correct_predictions/float(len(y_test))))
