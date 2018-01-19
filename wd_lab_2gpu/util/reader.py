#!/usr/bin/env python 
# read csv return a generator
import tensorflow as tf
import glob
import sklearn
import pandas as pd
import numpy as np

COLUMNS = ['label', 'browser', 'city', 'creativeid', 'iid_clked', 'iid_imp', 'itemtype', 'network', 'operation', 'os', 'province', 'psid_abs', 'pvday', 'pvhour', 'read', 'sessionid', 'source', 'userid']


class reader:

    def __init__(self,train_dir, test_dir, columns, numeric_col, batch_size, shuffle=True):

        self._train_dir = train_dir
        self._test_dir  = test_dir
        self._columns   = columns
        self._numeric_col = numeric_col
        self._batch_size = batch_size
        self._train_iter = self.get_train_iter()
        self._test_iter = self.get_test_iter()

    def get_train_iter(self):

        ifiles = self.get_files(self._train_dir)
        iter_data = self.read_data(ifiles, self._columns, self._numeric_col, self._batch_size)
        return iter_data

    def get_test_iter(self):

        ifiles = self.get_files(self._test_dir)
        iter_data = self.read_data(ifiles, self._columns, self._numeric_col, self._batch_size)
        return iter_data

    def get_train_batch(self, emb_col, onehot_col, numeric_col, wide_col):

        # read batch
        dataIter = self._train_iter
        data  = dataIter.next()

        emb_data = data[emb_col].values

        if numeric_col:
            numeric_value = data[numeric_col]
        else:
            numeric_value = None

        onehot_data = data[onehot_col].values
        wide_data = np.reshape(data[wide_col].values, (-1,len(wide_col)))
        label_data = np.reshape(data['label'].values,(-1,1))
        # psid_data
        # psid_data = np.reshape(data['psid_abs'].values,(-1,1))

        return emb_data, onehot_data, wide_data, label_data   # ,psid_data

    def get_test_batch(self, emb_col, onehot_col, numeric_col, wide_col):

        # read batch
        iter_data = self._test_iter 
        data  = iter_data.next()

        emb_data = data[emb_col].values

        if numeric_col:
            numeric_value = data[numeric_col]
        else:
            numeric_value = None

        onehot_data = data[onehot_col].values
        wide_data = np.reshape(data[wide_col].values, (-1,len(wide_col)))
        label_data = np.reshape(data['label'].values,(-1,1))
        # psid_data
        #psid_data = np.reshape(data['psid_abs'].values,(-1,1))

        return emb_data, onehot_data, wide_data, label_data #,psid_data


    def get_files(self, file_dir, shuffle = True):
    
        files = glob.glob(file_dir + "/part*")
        if shuffle:
            files = sklearn.utils.shuffle(files)       
        #files = tf.train.match_filenames_once(file_dir + "/part*")
        #print("[reader] files:")
        #print(files)
        return files
    
    
    def read_data(self, ifile, COLUMNS=COLUMNS,numeric_col=None,batch_size=30000):
    
        i = 0
        # select a file from ifile
        ind = i % len(ifile)
        oneFile = ifile[ind]
    
        print("[reader] read file: {} ".format(oneFile))
        reader = pd.read_csv(oneFile, header = None, names = COLUMNS, index_col = False, na_filter=True, dtype = np.str, iterator = True) 
    
        while True:
    
            try:
                data = reader.get_chunk(batch_size)
            except StopIteration:
                print "Iteration is stopped."
                i = i + 1
    
                # select a file from ifile
                ind = i % len(ifile)
                oneFile = ifile[ind]
                print("[reader] read file: {} ".format(oneFile))
                reader = pd.read_csv(oneFile, header = None, names = COLUMNS, index_col = False, na_filter=True, dtype = np.str, iterator = True) 
    
                continue
                    
            #data = pd.read_csv(oneFile, header = None, names = COLUMNS, index_col = False, na_filter=True, dtype = np.str) 
            #print("[reader] file shape: {} {}".format(data.shape[0], data.shape[1]))
    
            if numeric_col:
                string_col = [ col for col in COLUMNS if col not in numeric_col] 
                data[string_col] = data[string_col].fillna("missing")
                data[numeric_col] = data[numeric_col].fillna("0")
    
            else:
                string_col = COLUMNS
                data[string_col] = data[string_col].fillna("missing")
    
            yield data
    
    
    #def from_csv_file(ifile, batch_size, read_threads, shuffle = True, allow_smaller_final_batch = False):
    #    '''
    #       read csv file 
    #    '''
    #
    #    filename_queue = tf.train.string_input_producer(ifile, num_epochs = 1, shuffle = shuffle)
    #
    #    reader = tf.TextLineReader(skip_header_lines = None, name = "TextLineReader")
    #    _, example = reader.read(filename_queue, name = "readQueue")
    #    
    #    example_batch = tf.train.batch(example, batch_size = batch_size, capacity = batch_size * 5, allow_smaller_final_batch = allow_smaller_final_batch)
    #
    #    #return example    
    #
    #    return example_batch
    
if __name__ == "__main__":
    data_reader = reader("/data/new/dis_with_wide/train", "/data/new/dis_with_wide/test", ['label', 'browser', 'city', 'creativeid', 'iid_clked', 'iid_imp', 'itemtype', 'network', 'operation', 'os', 'province', 'psid_abs', 'pvday', 'pvhour', 'read', 'sessionid', 'source', 'userid'], None, 50000)
    emb_data, onehot_data, wide_data, label_data = data_reader.get_train_batch(['creativeid'], ['sessionid'], None, ['iid_imp'])
    print(emb_data)
    emb_data, onehot_data, wide_data, label_data = data_reader.get_test_batch(['creativeid'], ['sessionid'], None, ['iid_imp'])
    print(emb_data)

