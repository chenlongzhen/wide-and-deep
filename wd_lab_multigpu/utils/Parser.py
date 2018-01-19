#encoding=utf-8
import tensorflow as tf

import argparse
import collections
import yaml
from tensorflow.python.ops.sparse_ops import _sparse_cross_hashed
from process import get_onehot,_process_list_column
#tfe.enable_eager_execution()
import time
import numpy as np
from datetime import datetime
import os

class ConfParser:

    def __init__(self, conf_path):
        self.YAML_CONFIG = yaml.load(open(conf_path))
        self.COLUMNS = self.YAML_CONFIG['using_features_dl']
        self.FEATURE_CONF = self.YAML_CONFIG['features_conf']
        self.CARTESIAN_CROSS = self.YAML_CONFIG['cartesian_cross']
        self.LABEL_COLUMN = "label"
        self.POSTERIOR_PREFIX = ["poPV", "poCLK", "poCTR", "poCOEC_st", "poCOEC_em"]
        self.CARTESIAN_CROSS_DICT = {}


        print(self.COLUMNS)
        self.append_posterior()
        self.append_rounding()
        print(self.COLUMNS)

        # get csv header
        header = self.build_csv_header()

        # Order is important for the csv-readers, so we use an OrderedDict here.
        self.defaults = collections.OrderedDict(header)  # pyformat: disable

        self.types = collections.OrderedDict((key, type(value[0]))
                                        for key, value in self.defaults.items())
        print(self.types)

        self.PARSED_COLUMNS = [key for key, value in self.defaults.items()]
        print(self.PARSED_COLUMNS)

        self.get_onehot = get_onehot
        self._process_list_column = _process_list_column


    def _append_element(self,feat_name, feat_conf=None):
        self.COLUMNS.append(feat_name)
        self.FEATURE_CONF[feat_name] = {
            "feature_type": 'sparse',
            "model_type": 'deep'
        } if feat_conf is None else feat_conf


    def _replace_element(self,feat_name, new_name=None):
        if new_name:
            self.COLUMNS.append(new_name)
            self.FEATURE_CONF[new_name] = self.FEATURE_CONF[feat_name]
        self.COLUMNS.remove(feat_name)
        self.FEATURE_CONF.pop(feat_name)


    def append_posterior(self):
        posterior_days = [3]
        if self.YAML_CONFIG['if_posteriori_pv'] == 1:
            feats = [k for k, v in self.YAML_CONFIG['features_conf'].items() if v.get('post_pv', 0) == 1 and k in self.COLUMNS]
            for i in feats:
                for j in posterior_days:
                    self._append_element("poPV_" + i + "_" + str(j), self.YAML_CONFIG['features_conf']['poPV'])
        if self.YAML_CONFIG['if_posteriori_clk'] == 1:
            feats = [k for k, v in self.YAML_CONFIG['features_conf'].items() if v.get('post_clk', 0) == 1 and k in self.COLUMNS]
            for i in feats:
                for j in posterior_days:
                    self._append_element("poCLK_" + i + "_" + str(j), self.YAML_CONFIG['features_conf']['poCLK'])
        if self.YAML_CONFIG['if_posteriori_ctr'] == 1:
            feats = [k for k, v in self.YAML_CONFIG['features_conf'].items() if v.get('post_ctr', 0) == 1 and k in self.COLUMNS]
            for i in feats:
                for j in posterior_days:
                    self._append_element("poCTR_" + i + "_" + str(j), self.YAML_CONFIG['features_conf']['poCTR'])
        if self.YAML_CONFIG['if_posteriori_coec'] == 1:
            feats = [k for k, v in self.YAML_CONFIG['features_conf'].items() if v.get('post_coec', 0) == 1 and k in self.COLUMNS]
            for i in feats:
                for j in posterior_days:
                    if self.YAML_CONFIG['bias_method'] == 'STAT':
                        self._append_element("poCOEC_st_" + i + "_" + str(j), self.YAML_CONFIG['features_conf']['poCOEC_st'])
                    elif self.YAML_CONFIG['bias_method'] == 'EM':
                        self._append_element("poCOEC_em_" + i + "_" + str(j), self.YAML_CONFIG['features_conf']['poCOEC_em'])


    def append_rounding(self):
        tmp_d = [fea for fea in self.COLUMNS if
                 fea in self.FEATURE_CONF and 'use_rounding' in self.FEATURE_CONF[fea] and self.FEATURE_CONF[fea]['use_rounding'] == 1]
        for item in tmp_d:
            if not item.startswith('dl_'):
                self._replace_element(item, 'dl_' + item)


    def append_cartesian_cross(self):
        for item in self.CARTESIAN_CROSS:
            d = sorted(item.split("&"))
            if len(d) == 2 and d[0] in self.COLUMNS and d[1] in self.COLUMNS:
                self._append_element(d[0] + "&" + d[1])


    # get header default
    def _check_config(self, source, attr, legal_list=None, default=None):
        if source is None or attr not in source or (legal_list is not None and source[attr] not in legal_list):
            return default
        return source[attr]


    def build_csv_header(self):
        record_defaults = [('label', [0])]
        for fea in sorted(self.COLUMNS):
            feature_type = self.FEATURE_CONF[fea]['feature_type']
            # wide side setting String !
            if self.FEATURE_CONF[fea]['model_type'] == 'wide':
                record_defaults.append((fea, ["missing"]))
                # print ('[read_csv_file] append record = %s, type = %s' % (fea, 'tf.string'))
                continue #TODO wide on gpu

            if feature_type == 'sparse' or feature_type == 'multi_sparse':
                if 'feature_sparse' in self.FEATURE_CONF[fea] and self.FEATURE_CONF[fea]['feature_sparse'] == 'integer':
                    # print ('[read_csv_file] append record = %s, type = %s' % (fea, 'tf.int64'))
                    record_defaults.append((fea, [0.0]))
                else:
                # print ('[read_csv_file] append record = %s, type = %s' % (fea, 'tf.string'))
                        record_defaults.append((fea, ["missing"]))


            elif feature_type == 'real':
                fea_dim = self._check_config(self.FEATURE_CONF[fea], 'feature_dimension', default=1)
                # print ('[read_csv_file] append record = %s, type = %s, dim = %d' % (fea, 'tf.float32', fea_dim))
                if fea_dim == 1:
                    record_defaults.append((fea, [0.0]))
                else:
                    for i in range(fea_dim):
                        record_defaults.append((fea + "$" + str(i), [0.0]))
        return record_defaults


    def build_columns(self, items, emb_dict):
        wide_columns = []
        deep_columns = []
        print("===")
        print(self.FEATURE_CONF)
        # input assignment
        for fea in self.PARSED_COLUMNS:
            fea_config = self._check_config(self.FEATURE_CONF, fea.split("$")[0])  # if fea is real vec, split by '$'
            if fea_config is None:
                print("[build_estimator] incorrect input %s: no feature_conf." % (str(fea)))
                continue
            feature_type = self._check_config(fea_config, 'feature_type', legal_list=['sparse', 'multi_sparse', 'real'])
            model_type = self._check_config(fea_config, 'model_type', legal_list=['wide', 'deep'])
            layer = None
            # assign input type.

            if model_type == 'wide':
                if feature_type == 'sparse':
                    # wide 必须hash
                    feature_sparse = self._check_config(fea_config, 'feature_sparse', legal_list=['hash'], default='hash')
                    bucket_size = self._check_config(fea_config, 'bucket_size', default=1024)
                    print("[build_estimator] add sparse_column_with_hash_bucket, fea = %s, hash_bucket_size = %d" % (
                        str(fea), bucket_size))
                    _, onehot_emb = self.get_onehot(items[fea], bucket_size)
                    wide_columns.append(
                        tf.reshape(onehot_emb, [1, bucket_size])
                    )

                elif feature_type == 'multi_sparse':
                    # wide 必须hash
                    feature_sparse = self._check_config(fea_config, 'feature_sparse', legal_list=['hash'], default='hash')
                    bucket_size = self._check_config(fea_config, 'bucket_size', default=1024)
                    print("[build_estimator] add multi_sparse_column_with_hash_bucket, fea = %s, hash_bucket_size = %d" % (
                        str(fea), bucket_size))
                    wide_columns.append(
                        self._process_list_column(tf.reshape(items[fea], [-1]), bucket_size)
                    )
                else:
                    raise ("wide build column error!")

            elif model_type == 'deep':

                model_feed_type = self._check_config(fea_config, 'model_feed_type', legal_list=['embedding', 'onehot', 'real'],
                                                default='onehot')
                bucket_size = self._check_config(fea_config, 'bucket_size', default=1024)

                if model_feed_type == 'embedding':
                    dimension = self._check_config(fea_config, 'dimension', default=32)
                    print("[build_estimator] add embedding_column, fea = %s, hash_bucket_size = %d, dimension = %d" % (
                        str(fea), bucket_size, dimension))
                    onehot_value, onehot_emb = self.get_onehot(items[fea], bucket_size)
                    lookup_emb = tf.nn.embedding_lookup(emb_dict[fea], onehot_value)
                    deep_columns.append(
                        tf.reshape(lookup_emb, [1, dimension])
                    )


                elif model_feed_type == 'onehot':
                    print("[build_estimator] add one_hot_column, fea = %s, hash_bucket_size = %d" % (str(fea), bucket_size))
                    onehot_value, onehot_emb = self.get_onehot(items[fea], bucket_size)
                    deep_columns.append(
                        tf.reshape(onehot_emb, [1, bucket_size])
                    )

                elif model_feed_type == 'real':
                    print("[build_estimator] add real_valued_column, fea = %s" % (str(fea)))
                    deep_columns.append(
                        tf.reshape(items[fea], [1, 1])
                    )

                else:
                    print("[build_estimator] incorrect input %s: illegal model_feed_type." % (str(fea)))

            else:
                print("[build_estimator] incorrect input %s: illegal model_type" % (str(fea)))

        # crossing
        cross_sparse = _sparse_cross_hashed(wide_columns, num_buckets=3000000)

        fix_cross_sparse = tf.SparseTensor(
            indices=cross_sparse.indices,
            values=cross_sparse.values,
            dense_shape=cross_sparse.dense_shape)


        deep_line = tf.concat(deep_columns, axis=1)
        deep_line = tf.reshape(deep_line, shape=[-1])

        wide_line = fix_cross_sparse

        # print some info for the columns registered to different types
        print("[build_estimator] wide columns: %d" % (len(wide_columns)))
        # print(wide_columns)
        print("[build estimator] deep columns: %d" % (len(deep_columns)))
        # print(deep_columns)
        print("[build estimator] cross_sparse:")
        # print(cross_sparse) # crossed wide columns

        # return wide_line, deep_line
        print("[deep_line]")
        # print(deep_line)
        return deep_line


    def dataset(self, emb_dict, y_name="label", batch_size=1, filenames=["/data/wandd/demo_data/tiny_tr/part-00000"]):
        """Load the imports85 data as a (train,test) pair of `Dataset`.
        Each dataset generates (features_dict, label) pairs.
        Args:
          y_name: The name of the column to use as the label.
          train_fraction: A float, the fraction of data to use for training. The
              remainder will be used for evaluation.
        Returns:
          A (train,test) pair of `Datasets`
        """

        # Define how the lines of the file should be parsed
        def decode_line(line):
            """Convert a csv line into a (features_dict,label) pair."""
            # Decode the line to a tuple of items based on the types of
            # csv_header.values().
            items = tf.decode_csv(line, list(self.defaults.values()))

            # Convert the keys and items to a dict.
            pairs = zip(self.defaults.keys(), items)
            features_dict = dict(pairs)

            # Remove the label from the features_dict
            label = features_dict.pop(y_name)

            deep_line = self.build_columns(items=features_dict, emb_dict=emb_dict)


            return deep_line, label

        base_dataset = (tf.data
                        # Get the lines from the file.
                        .TextLineDataset(filenames)
                        .map(decode_line)
                        )

        # batched_dataset = base_dataset.batch(batch_size)

        # iterator = batched_dataset.make_initializable_iterator()
        # next_element = iterator.get_next()
        return base_dataset
        # Decode each line into a (features_dict, label) pair.
        # .map(decode_line))

        # Do the same for the test-set.
        # test = (base_dataset.filter(in_test_set).cache().map(decode_line))