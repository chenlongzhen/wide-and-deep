#encoding=utf-8
import tensorflow as tf
from tensorflow.python.ops.sparse_ops import  _sparse_cross_hashed


# embedding func ======================================
def embedding(num_size, num_dim, name="embedding"):
    '''
       initialize embeding matrix
    '''
    with tf.name_scope(name) as scope:
        # Declare all variables we need.
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / num_dim
        emb = tf.Variable(
            tf.random_uniform(
                [num_size, num_dim], -init_width, init_width),
            name=name + "emb")
        return emb


def multiEmbedding(embedding_para, name="multiEmbedding"):
    '''
       initialize many embeding matrix
    '''
    with tf.name_scope(name) as scope:
        emb_list = []
        for num_size, dim, dtype, col_name in embedding_para:
            emb_list.append(embedding(num_size, dim))

        return emb_list


def get_emb(input_value, emb_list, embedding_para, name="look_up_embedding"):
    '''
       hash string value and lookup embeding matrix
    '''
    with tf.name_scope(name) as scope:
        lookup_emb_list = []
        for i, emb in enumerate(emb_list):
            para = embedding_para[i]
            bucket_size = para[0]
            col_type = para[2]

            if col_type == "string":
                hash_value = tf.string_to_hash_bucket_fast(input_value[:, i], bucket_size)
            else:
                hash_value = tf.string_to_number(input_value[:, i], out_type=tf.int32)

            lookup_emb = tf.nn.embedding_lookup(emb, hash_value)
            lookup_emb_list.append(lookup_emb)

        return tf.concat(lookup_emb_list, axis=1)


# =======onehot func
def get_onehot(input_value, onehot_para, name="get_onehot"):
    '''
       onehot string or int value
    '''
    with tf.name_scope(name) as scope:

        if input_value.dtype == "string":
            hash_value = tf.string_to_hash_bucket_fast(input_value, onehot_para)
        else:
            hash_value = tf.string_to_number(input_value, out_type=tf.int32)

        onehot_emb = tf.one_hot(hash_value, onehot_para)

        return hash_value, onehot_emb

        # ========numeric func


def get_numeric(input_value, numeric_para, name="get_numeric"):
    '''
       onehot string or int value
    '''
    with tf.name_scope(name) as scope:
        numeric_list = []
        for i, numeric_para in enumerate(numeric_para):
            _ = numeric_para[0]  # no use now; maybe for bounding
            col_type = numeric_para[1]

            num_value = tf.string_to_number(input_value[:, i], out_type=tf.float32)
            num_value = tf.reshape(num_value, [-1, 1])

            numeric_list.append(num_value)

        if len(numeric_list) > 1:
            return tf.concat(numeric_list, axis=1)
        else:
            return numeric_list[0]





# ======= split
def _process_list_column(list_column, vocab_size):
    '''
       stringlist col  to dense tensor
       string col to onehot  tensor
    '''
    sparse_strings = tf.string_split(list_column, delimiter='##')
    sparse_ints = tf.SparseTensor(
        indices=sparse_strings.indices,
        values=tf.string_to_hash_bucket_fast(sparse_strings.values, vocab_size),
        dense_shape=sparse_strings.dense_shape)
    # return tf.cast(tf.sparse_to_indicator(sparse_ints, vocab_size = vocab_size), tf.float32), sparse_ints
    return sparse_ints


def process_wide(input_value, wide_para, name="wide_process"):
    '''
        process_wide features (split onehot)
    '''

    with tf.name_scope(name) as scope:
        wide_list = []
        cross_list = []
        fix_add_value = 0  # for concat
        for i, wide_para in enumerate(wide_para):
            depth = wide_para[0]
            col_type = wide_para[1]

            if col_type == "stringList" or True:
                sparse_value = _process_list_column(input_value[:, i], depth)

                fix_sparse_value = tf.SparseTensor(
                    indices=sparse_value.indices,
                    values=sparse_value.values + fix_add_value,
                    dense_shape=sparse_value.dense_shape)

                # wide_value, sparse_value = _process_list_column(input_value[:,i], depth)
                # wide_value = tf.reshape(wide_value, [-1, depth])
                print("wide_value")
                print(fix_sparse_value)

                wide_list.append(sparse_value)
                cross_list.append(sparse_value)

                fix_add_value = depth
                print("add {}".format(fix_add_value))
                # else:
                #     wide_value = tf.string_to_hash_bucket_fast(input_value[:,i], depth)
                #     print(wide_value)

                #     1
                #     onehot_emb = tf.one_hot(wide_value,depth, dtype = tf.float32)
                #     print(onehot_emb)
                #     wide_list.append(onehot_emb)
        # crossing
        cross_sparse = _sparse_cross_hashed(cross_list, num_buckets=3000000)
        fix_cross_sparse = tf.SparseTensor(
            indices=cross_sparse.indices,
            values=cross_sparse.values + 40000,
            dense_shape=cross_sparse.dense_shape)

        print(cross_sparse)

        # cross_value = tf.cast(tf.sparse_to_indicator(cross_sparse, vocab_size = 500000), tf.float32)
        # cross_value = tf.reshape(cross_value, [-1, 500000])
        wide_list.append(cross_sparse)
        print(wide_list)

        wide_sparse = tf.sparse_concat(sp_inputs=wide_list, axis=1)

        # wide_indicator = tf.cast(tf.sparse_to_indicator(wide_sparse, vocab_size = 140000), tf.float32)
        # wide_indicator = tf.reshape(wide_indicator, [-1,140000])
        return wide_sparse
