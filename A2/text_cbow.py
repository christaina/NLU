import tensorflow as tf
import numpy as np


class TextCBOW(object):
    """
    Using CBOW to clasify text
    """

    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # ignore padding
            flat_mask = tf.greater(self.input_x,0)
            flat_mask = tf.cast(flat_mask,tf.float32)
            self.mask = tf.expand_dims(flat_mask,[-1])
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.emb_masked = tf.mul(self.embedded_chars,self.mask)

        self.emb_masked = tf.mul(self.embedded_chars,self.mask)
        self.projection = tf.reduce_mean(self.emb_masked,reduction_indices=1)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[embedding_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.projection, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
