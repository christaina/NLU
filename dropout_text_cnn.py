import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def conv_wrapper(self,x,W,b,strides=1):
        conv = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding="VALID", name="conv")
        conv = tf.nn.bias_add(conv,b)
        return tf.nn.relu(conv, name="relu")

    def pool_wrapper(self,x,k,name="pool"):
        pooled = tf.nn.max_pool(
            x,
            ksize=[1, k, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name=name)
        return pooled


    def convpool(self,conv_input, filter_sizes, embedding_size, num_filters,pool_k="shape"):
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                h = self.conv_wrapper(conv_input,W,b,strides=1)
                # Maxpooling over the outputs
                pooled = self.pool_wrapper(h, k=pool_k-filter_size+1)
                pooled_outputs.append(pooled)
        return pooled_outputs

    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes,filter_sizes_2, num_filters,num_filters_2,
      l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout (which you need to implement!!!!)
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)
        # scaling weights when evaluating
        self.dropout_train_prob = tf.placeholder(tf.float32)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.wshape = W.get_shape()
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        
        pooled_outputs = self.convpool(self.embedded_chars_expanded, filter_sizes, \
                embedding_size, num_filters,pool_k=sequence_length)
        self.h_pool = tf.concat(3, pooled_outputs)
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add dropout
        self.h_drop = tf.nn.dropout(self.h_pool_flat,self.dropout_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                # use initialization from hes paper
                initializer=tf.contrib.layers.variance_scaling_initializer())

            # scale weights when testing
            if (self.dropout_prob != self.dropout_train_prob):
                W = tf.mul(W,self.dropout_train_prob)
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
