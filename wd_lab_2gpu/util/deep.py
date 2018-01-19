#!/bin/env python
import tensorflow as tf
from tensorflow.python.ops import nn
def deep_net(deep_input,mode):
    '''
    deep nets
    '''

    with tf.variable_scope('deep_model', values = (deep_input,)) as dnn_hidden_scope:

        with tf.variable_scope('dnn_hidden_1', values = (deep_input,)) as dnn_hidden_scope:
            deep_hidden_1 = tf.layers.dense(
                deep_input, 1024, activation = nn.relu)
            deep_hidden_1 = tf.layers.dropout(
                deep_hidden_1, rate = 0.9,
                training = mode == "train")
            if mode == "train":
                tf.summary.scalar("%s/fraction_of_zero_values" % dnn_hidden_scope.name, nn.zero_fraction(deep_hidden_1))
                tf.summary.histogram("%s/activation" % dnn_hidden_scope.name, deep_hidden_1) 

        with tf.variable_scope('dnn_hidden_2', values = (deep_hidden_1,)) as dnn_hidden_scope:
            #deep_hidden_2 = tf.contrib.layers.fully_connected(
            #    deep_hidden_1, 512, activation_fn = nn.relu, variables_collections = ['deep'])
            deep_hidden_2 = tf.layers.dense(
                deep_hidden_1, 512, activation = nn.relu)
            deep_hidden_2 = tf.layers.dropout(
                deep_hidden_2, rate = 0.9,
                training = mode == "train")
            if mode == "train":
                tf.summary.scalar("%s/fraction_of_zero_values" % dnn_hidden_scope.name, nn.zero_fraction(deep_hidden_2))
                tf.summary.histogram("%s/activation" % dnn_hidden_scope.name, deep_hidden_2)
        # deep_hidden_2_merge = tf.concat([deep_hidden_2, deep_input_psid], 1)

        with tf.variable_scope('dnn_hidden_3', values = (deep_hidden_2,)) as dnn_hidden_scope:
            deep_hidden_3 = tf.layers.dense(
                deep_hidden_2, 256, activation = nn.relu)
            deep_hidden_3 = tf.layers.dropout(
                deep_hidden_3, rate = 0.9,
                training = mode == "train")
            if mode == "train":
                tf.summary.scalar("%s/fraction_of_zero_values" % dnn_hidden_scope.name, nn.zero_fraction(deep_hidden_3))
                tf.summary.histogram("%s/activation" % dnn_hidden_scope.name, deep_hidden_3)

        #deep_hidden_3_con = tf.concat([deep_hidden_3,deep_input],1) # high way
        #hash_value = tf.string_to_hash_bucket_fast(psid_value[:,0],42)
        #psid_value = tf.one_hot(hash_value, 42)
        #deep_hidden_3_con = tf.concat([deep_hidden_3,psid_value],1)
        with tf.variable_scope('dnn_logits', values = (deep_hidden_3,)) as dnn_logits_scope:
            deep_logits = tf.layers.dense(
                deep_hidden_3, 1, activation = None, bias_initializer = None)
            if mode == "train":
                tf.summary.scalar("%s/fraction_of_zero_values" % dnn_logits_scope.name, nn.zero_fraction(deep_logits))
                tf.summary.histogram("%s/activation" % dnn_logits_scope.name, deep_logits)


    return deep_logits

def wide_net_old(wide_input, mode):

    with tf.variable_scope('wide_model', values = (wide_input,)) as dnn_hidden_scope:
        wide_logits = tf.layers.dense(
            wide_input, 1, activation = None)
        if mode == "train":
            tf.summary.scalar("%s/fraction_of_zero_values" % dnn_hidden_scope.name, nn.zero_fraction(wide_logits))
            tf.summary.histogram("%s/activation" % dnn_hidden_scope.name, wide_logits) 
    return wide_logits

def wide_net(wide_input, w_number ,mode):

    with tf.variable_scope('wide_model', values = (wide_input,)) as dnn_hidden_scope:

        embeddings = tf.Variable(tf.truncated_normal(
            (w_number,),
            mean=0.0,
            stddev=1.0,
            dtype=tf.float32,
            seed=None,
            name="wide_init_w"
        ))
        bias = tf.Variable(tf.random_normal((1,)))
        
        wide_logits = tf.nn.embedding_lookup_sparse(embeddings, wide_input, None, combiner="sum") + bias

        if mode == "train":
            tf.summary.scalar("%s/fraction_of_zero_values" % dnn_hidden_scope.name, nn.zero_fraction(wide_logits))
            tf.summary.histogram("%s/activation" % dnn_hidden_scope.name, wide_logits) 
    return tf.reshape(wide_logits, [-1,1])
