#!/bin/env python

from tensorflow.python.tools import freeze_graph


prefix = "/data/new/wandd/wd_lab/model_test/"
input_graph_path = prefix + "widendeep.pbtxt"
input_saver_def_path = ""
input_binary = False
output_node_names = "prediction"
restore_op_name = "save"
filename_tensor_name = "save/Const:0"
clear_devices = True
input_meta_graph = prefix + "my-model.meta"
checkpoint_path = prefix + "my-model-300"
output_graph_filename= "./outGraph.pb"
freeze_graph.freeze_graph(
    input_graph_path, input_saver_def_path, input_binary, checkpoint_path,
    output_node_names, restore_op_name, filename_tensor_name,
    output_graph_filename, clear_devices, "")
