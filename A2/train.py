#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cbow import TextCBOW
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 10, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("cutoff", 300, "When to cut off the sentence (default: 50)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_float("lr", 1e-3, "Learning Rate")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("bigrams",True,"Use bigrams")
tf.flags.DEFINE_string("n","best_ceb545","Use bigrams")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

train_pos = './data/aclImdb/train_pos_top.txt'
train_neg = './data/aclImdb/train_neg_top.txt'
bigram_pos = './data/aclImdb/train_pos_top.txt'
bigram_neg = './data/aclImdb/train_neg_top.txt'


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels('data/aclImdb/train_pos_top.txt','data/aclImdb/train_neg_top.txt')

# Cut off sentences
x_text = [(" ").join(x.split(" ")[:FLAGS.cutoff]) for x in x_text]
# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
vocab_size = len(vocab_processor.vocabulary_)
print(x.shape)

if (FLAGS.bigrams==True):
    bigrams_features = data_helpers.load_bigram_feats('./data/aclImdb/train_pos_bi.txt','./data/aclImdb/train_neg_bi.txt')
    bigrams_list = (data_helpers.load_bigrams_file("./data/aclImdb/bigrams.txt"))
    bigrams_list = [" ".join(i) for i in bigrams_list]
    #bigrams_list = set(bigrams_list)
    vocab_size = vocab_size+len(bigrams_list)+1
    bigram_st = len(vocab_processor.vocabulary_)+1
    max_bigram_length = max([len(bi) for bi in bigrams_features])
    bigrams_mat = np.empty([len(bigrams_features),max_bigram_length])
    print(bigrams_mat.shape)

    bigrams_dict = {}

    for i,r in enumerate(bigrams_features):
        for j,w in enumerate(r):
            if w == "":
                bigrams_mat[i][j]=0
            #if w not in set(bigrams_dict.keys()):
            #    bigrams_dict[w] = bigram_st
            #    bigram_st+=1
            else:

                bigrams_mat[i][j]=bigrams_list.index(w)+bigram_st
    x = np.concatenate((x,bigrams_mat),axis=1)
    #x = np.vstack((x,bigrams_mat))

    # create own dictionary on these
    # once have values, append to end of transformed data
    # update vocab length values

val,train = data_helpers.split_data(x,y)
x_train,y_train = train
x_dev,y_dev = val


print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = TextCBOW(
            sequence_length=x_train.shape[1],
            num_classes=2,
            vocab_size=vocab_size,
            #vocab_size=bigram_st,
            embedding_size=FLAGS.embedding_dim,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
       
        fn = FLAGS.n
        timestamp = str(int(time.time()))
        if FLAGS.n=='time':
            fn = timestamp
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", fn))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", model.loss)
        acc_summary = tf.scalar_summary("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
