#!/usr/bin/env python

import tensorflow as tf



class emmbedding:

    def __init__(self,num_buckets,num_dim,name=""):
        self._emb = embbeding(num_buckets,num_dim)
        self._num_buckets = num_buckets
        self._num_dim = num_dim
        self._name = name
        

    def embbeding(self):

        # Declare all variables we need.
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / emb_dim
        emb = tf.Variable(
            tf.random_uniform(
                [self._num_buckets, self._num_dim], -init_width, init_width),
            name=self._name + "_emb")
        return emb 


    def get_emb(self):
        return self._emb

#    def hash_looklookup(input_string,name=""):
#        '''
#            input_string: A Tensor of type string. The strings to assign a hash bucket.
#        '''
#        value = tf.string_to_hash_bucket_fast(input_string, self._num_buckets, self._name=name + "_hash_fast")
#        lookup_emb = tf.nn.embedding_lookup(self._emb, value)
#        return lookup_emb
        
    
