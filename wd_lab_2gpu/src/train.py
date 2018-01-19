#!/bin/env python 

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import sys, os
sys.path.append("..")
from util.deep import deep_net,wide_net
from util.reader import *
from tqdm import tqdm
from tensorflow.python.ops.sparse_ops import  _sparse_cross_hashed

embedding_para = [(100,32,"string","creativeid"),(100,32,"string","sessionid"),(100,32,"string","userid")]
onehot_para = [(41,"string","psid_abs"),(4, "string", "operation"),(20,"string","itemtype"),(10,"string","pvday"), (24, "int", "pvhour"), (15, "string", "network"), (2, "int","read"), (25, "string", "browser"), (2, "string", "source"), (12, "string", "os"), (36, "string", "province")]

numeric_para = None

wide_para = [(20000,"string","iid_imp"),(20000,"stringList","iid_clked")]

def get_columns(paras):

    column = [col[-1] for col in paras]
    return column

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
        for num_size,dim,dtype,col_name in embedding_para:
            emb_list.append(embedding(num_size,dim))

        return emb_list

def get_emb(input_value,emb_list,embedding_para, name = "look_up_embedding"):
    '''
       hash string value and lookup embeding matrix
    '''
    with tf.name_scope(name) as scope:
        lookup_emb_list = []
        for i,emb in enumerate(emb_list):
            para = embedding_para[i]
            bucket_size = para[0]  
            col_type = para[2]  

            if col_type == "string":
                hash_value = tf.string_to_hash_bucket_fast(input_value[:,i],bucket_size)
            else:
                hash_value = tf.string_to_number(input_value[:,i],out_type = tf.int32)

            lookup_emb = tf.nn.embedding_lookup(emb,hash_value)
            lookup_emb_list.append(lookup_emb)

        return tf.concat(lookup_emb_list,axis = 1)
    

# =======onehot func
def get_onehot(input_value,onehot_para, name = "get_onehot"):

    '''
       onehot string or int value
    '''
    with tf.name_scope(name) as scope:
        onehot_list = []
        for i,onehot_para in enumerate(onehot_para):
            depth = onehot_para[0]  
            col_type = onehot_para[1]  

            if col_type == "string":
                hash_value = tf.string_to_hash_bucket_fast(input_value[:,i],depth)
            else:
                hash_value = tf.string_to_number(input_value[:,i],out_type = tf.int32)

            onehot_emb = tf.one_hot(hash_value,depth)
            onehot_list.append(onehot_emb)

        return tf.concat(onehot_list, axis = 1)

# ========numeric func
def get_numeric(input_value,numeric_para, name = "get_numeric"):

    '''
       onehot string or int value
    '''
    with tf.name_scope(name) as scope:
        numeric_list = []
        for i,numeric_para in enumerate(numeric_para):
            _ = numeric_para[0]   # no use now; maybe for bounding
            col_type = numeric_para[1]  

            num_value = tf.string_to_number(input_value[:,i], out_type = tf.float32)
            num_value = tf.reshape(num_value, [-1,1])

            numeric_list.append(num_value)

        if len(numeric_list) > 1:
            return tf.concat(numeric_list, axis = 1)
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
            indices = sparse_strings.indices,
            values = tf.string_to_hash_bucket_fast(sparse_strings.values,vocab_size),
            dense_shape = sparse_strings.dense_shape)
    #return tf.cast(tf.sparse_to_indicator(sparse_ints, vocab_size = vocab_size), tf.float32), sparse_ints
    return sparse_ints

def process_wide(input_value, wide_para, name = "wide_process"):
    '''
        process_wide features (split onehot)
    '''

    with tf.name_scope(name) as scope:
        wide_list = []
        cross_list = []
        fix_add_value = 0 # for concat  
        for i,wide_para in enumerate(wide_para):
            depth = wide_para[0]  
            col_type = wide_para[1]  

            if col_type == "stringList" or True:
                sparse_value = _process_list_column(input_value[:,i], depth)

                fix_sparse_value = tf.SparseTensor(
                    indices = sparse_value.indices,
                    values = sparse_value.values + fix_add_value,
                    dense_shape = sparse_value.dense_shape)

                #wide_value, sparse_value = _process_list_column(input_value[:,i], depth)
                #wide_value = tf.reshape(wide_value, [-1, depth])
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
        cross_sparse = _sparse_cross_hashed(cross_list, num_buckets = 3000000)
        fix_cross_sparse = tf.SparseTensor(
                    indices = cross_sparse.indices,
                    values = cross_sparse.values + 40000,
                    dense_shape = cross_sparse.dense_shape)
                
        print(cross_sparse)

        #cross_value = tf.cast(tf.sparse_to_indicator(cross_sparse, vocab_size = 500000), tf.float32)
        #cross_value = tf.reshape(cross_value, [-1, 500000])
        wide_list.append(cross_sparse)
        print(wide_list)

        wide_sparse = tf.sparse_concat(sp_inputs = wide_list, axis = 1)

        #wide_indicator = tf.cast(tf.sparse_to_indicator(wide_sparse, vocab_size = 140000), tf.float32)
        #wide_indicator = tf.reshape(wide_indicator, [-1,140000])
        return wide_sparse


# ========
def build_wide(wide_value):

    with tf.device("/gpu:1"):
        wide_emb = process_wide(wide_value,wide_para)
        print("wide_emb")
        print(wide_emb)

        wide_logits = wide_net(wide_emb, 3040000,mode = "train")

        return  wide_logits


def build_deep(emb_value, onehot_value, numeric_value = None):

    deep_input_list = []

    # embedding op 
    
    emb_list = multiEmbedding(embedding_para)
    
    lookup_emb = get_emb(emb_value,emb_list, embedding_para)
    deep_input_list.append(lookup_emb)
    
    # onehot op
    onehot_emb = get_onehot(onehot_value,onehot_para)
    deep_input_list.append(onehot_emb)
    
    # numeric op
    if numeric_value:
        numeric_emb = get_numeric(numeric_value,numeric_para)
        deep_input_list.append(numeric_emb)
    
    
    # concat 
    deep_input = tf.concat(deep_input_list, axis = 1 ) 
    
    #deep net
    deep_logits = deep_net(deep_input,mode = "train")
    return deep_logits


####################################
# place holder
with tf.name_scope("INPUT") as scope, tf.device("/cpu:0"):
    emb_value = tf.placeholder(tf.string, shape = [None, len(embedding_para)], name = "emb_placeholder")
    onehot_value = tf.placeholder(tf.string, shape = [None, len(onehot_para)], name = "onehot_placeholder")
    if numeric_para :
        numeric_value = tf.placeholder(tf.string, shape = [None, len(numeric_para)], name = "numeric_placeholder")
    wide_value = tf.placeholder(tf.string, shape = [None,len(wide_para)], name = "wide_placeholder")
    #psid_value = tf.placeholder(tf.string, shape = [None,1], name = "psid_placeholder")
    label = tf.placeholder(tf.string, shape = [None, 1], name = "label")


# deep net 
with tf.name_scope("deep_logits") as scope:

    with tf.device("/gpu:0"):
        deep_logits = build_deep(emb_value, onehot_value,numeric_value = None)
        tf.summary.histogram(scope,deep_logits)


# wide net
# gpu is set in the func
with tf.name_scope("wide_logits") as scope:
    wide_logits =  build_wide(wide_value)
    tf.summary.histogram(scope,wide_logits)

# combine
with tf.device("/cpu:0"):

    logits = tf.add(deep_logits , wide_logits)
    tf.summary.histogram("logits",logits)

    # predict
    predictions = tf.sigmoid(logits, name='prediction')
    tf.summary.histogram('predictions', predictions)

    # train loss
    label = tf.string_to_number(label,out_type = tf.int32)
    label = tf.reshape(label, [-1,1])
    
    training_loss = tf.losses.sigmoid_cross_entropy(label, logits)
    tf.summary.scalar('cross_entropy', training_loss)


# OPT
with tf.device("/gpu:0"):
    deep_opt = tf.train.AdagradOptimizer(
        learning_rate = 0.5, name = "Adagrad"
                )
    
    train_op_deep = deep_opt.minimize(
        loss = training_loss,
        global_step=tf.train.get_global_step(),
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"deep_logits*") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"deep_model*")
    )
    
with tf.device("/gpu:1"):
    wide_opt = tf.train.FtrlOptimizer(
            learning_rate = min(0.01, 1.0 / len(wide_para)),
            learning_rate_power=-0.5,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0,
            name='Ftrl',
            accum_name=None,
            linear_name=None,
            l2_shrinkage_regularization_strength=0.0
        )
    
    train_op_wide = wide_opt.minimize(
             training_loss,
             global_step=tf.train.get_global_step(),
             var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"wide_logits*"),
             aggregation_method=None,
             colocate_gradients_with_ops=False,
             name=None,
             grad_loss=None
         )


train_ops = [train_op_deep, train_op_wide]
train_op = control_flow_ops.group(*train_ops)

print("TRAIN VARIBLES DEEP")
#print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"deep_model*") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"multiEmbedding*"))
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"deep_logits*") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"deep_model*"))
print("TRAIN VARIBLES WIDE")
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"wide_logits*"))

# metric
with tf.name_scope("metric") as scope, tf.device("/gpu:0"):
    auc_op = tf.metrics.auc(label,predictions)
    tf.summary.scalar(scope + "AUC", auc_op[0])
    
merged = tf.summary.merge_all()

def main(_):
#embedding_para = [(100,32,"string","creativeid"),(100,32,"string","clicked"),(100,32,"string","sessionid"),(100,32,"string","userid")]
#onehot_para = [(3,"string","dt"),(41,"string","psid_abs"),(4, "string", "operation"),(20,"string","itemtype"),(10,"string","pvday"), (24, "int", "pvhour"), (15, "string", "network"), (2, "int","read"), (25, "string", "browser"), (2, "string", "source"), (12, "string", "os"), (36, "string", "province")]
#
#numeric_para = []

#wide_para = [(20000,"string","iid_imp")]

    train_steps = 100
    test_steps = 100
    COLUMNS = ['label', 'browser', 'city', 'creativeid', 'iid_clked', 'iid_imp', 'itemtype', 'network', 'operation', 'os', 'province', 'psid_abs', 'pvday', 'pvhour', 'read', 'sessionid', 'source', 'userid']
    model_dir = "../model_test/"

    emb_col = get_columns(embedding_para)

    print("emb_col")
    print(emb_col)

    onehot_col = get_columns(onehot_para)
    print("onehot_col")
    print(onehot_col)

    if numeric_para:
        numeric_col = get_columns(numeric_para)
    else:
        numeric_col = None

    print("numeric_col")
    print(numeric_col)
    wide_col = get_columns(wide_para)

    print("wide_col")
    print(wide_col)


    # read data
    data_reader = reader("/data/new/dis_with_wide/train", "/data/new/dis_with_wide/test", COLUMNS, numeric_col, 10000)

    global mode
    # restore data >
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#    config.gpu_options.allocator_type = 'BFC'

    init = tf.initialize_all_variables()
    sm = tf.train.SessionManager()
    # try to find the latest checkpoint in my_checkpoint_dir, then create a session with that restored
    # if no such checkpoint, then call the init_op after creating a new session
    with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=model_dir, config = config) as sess:
        tf.train.write_graph(sess.graph_def, model_dir, 'widendeep.pbtxt')
    #with tf.Session(config = config) as sess:

        #tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        train_writer = tf.summary.FileWriter(model_dir + 'train',
                                                      sess.graph)
        test_writer = tf.summary.FileWriter(model_dir + 'test')

        for epoch in range(1000):
            #print("[INFO] step:{}".format(tf.global_step()))
            
            train_auc_list = []
            train_loss_list = []
            test_auc_list = []
            test_loss_list = []

            for step in range(train_steps):

                emb_data, onehot_data, wide_data, label_data = data_reader.get_train_batch(emb_col, onehot_col, numeric_col, wide_col)
                
                #merge_summary,  _ = sess.run([merged, train_op],feed_dict={emb_value: [["clz","3"],["1","2"]], onehot_value: [["clz","3"],["1","2"]] , numeric_value: [["1","3"],["1","2"]], label: [["1"],["1"]], wide_value: [["1","3"],["1","2"]]})            
                
                train_feed = {emb_value: emb_data, onehot_value: onehot_data,label: label_data, wide_value: wide_data}

                merge_summary,  _ , auc, loss= sess.run([merged, train_op, auc_op, training_loss],feed_dict=train_feed)
                train_writer.add_summary(merge_summary, (epoch+1) * train_steps +  (step+1) )

                train_auc_list.append(auc[0])
                train_loss_list.append(loss)
                
            # save 
            #saver.save(sess, model_dir + 'my-model', global_step=None)
            checkpoint_path = saver.save(sess, model_dir + 'my-model', global_step=(epoch+1) * train_steps +  (step+1), latest_filename="checkpoint_state")
            print("checkpoint_path: {}".format(checkpoint_path))


            # get train AUC
            mean_auc = np.mean(train_auc_list)
            mean_loss    = np.mean(train_loss_list)
            print("[TRAIN metic] range: {}, loss: {} ".format(epoch,mean_loss))
            print("[TRAIN metic] range: {}, auc: {} ".format(epoch,mean_auc))

            for step in range(test_steps):


                emb_data, onehot_data, wide_data, label_data = data_reader.get_test_batch(emb_col, onehot_col, numeric_col, wide_col)


                test_feed = {emb_value: emb_data, onehot_value: onehot_data,label: label_data, wide_value: wide_data}
                merge_summary, pred, loss, auc = sess.run([merged, predictions, training_loss, auc_op], feed_dict = test_feed)
                test_writer.add_summary(merge_summary, (epoch+1) * train_steps +  (step+1) )
                
                test_auc_list.append(auc[0])
                test_loss_list.append(loss)
            
            mean_auc = np.mean(test_auc_list)
            mean_loss    = np.mean(test_loss_list)
            print("[test metic] range: {}, loss: {} ".format(epoch,mean_loss))
            print("[test metic] range: {}, auc: {} ".format(epoch,mean_auc))

main(1)
