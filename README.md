# wide and deep 

wide and deep 代码结构：

part |功能 |代码位置
---|---|---
reader | 读取数据 | reader.py
processer| 数据embedding,onehot等| train.py
build graph|wide and deep | deep.py
freeze and load graph|固化模型/加载模型|freeze.py/loadGraph.py

git:https://git.corpautohome.com/gp_mb_ad_algo_test/wandd/tree/master/wd_lab

```
|-- model_dir
|-- src
|   `-- train.py
`-- util
    |-- __init__.py
    |-- deep.py
    |-- feature_processing.py
    |-- freeze.py
    |-- loadGraph.py
    |-- outGraph.pb
    |-- reader.py
```

## reader

读取本地训练测试数据喂给placeholder。主要使用pandas 生成迭代器分块读取数据，每次读取的数据量为一个batch。   同时对缺失数据进行填充，数值型数据填充为"0",字符型数据填充为"missing"。

主要步骤：
- 对文件夹下文件shuffle
- 将shuffle后文件顺序循环读取为迭代器
- 缺失值填充
- 每次调用从迭代器取一个batch

## processer

包含embedding，onehot,wide processing等。

### embedding
- 初始化embedding： func multiEmbedding
- 对输入数据hash／strig2number，再lookup：func get_emb
- 水平拼接每个特征提取的emb矩阵：func get_emb

### onehot
- 对输入数据hash／strig2number: func get_onehot
- 水平拼接每个特征提取的emb矩阵：func get_onehot
- concat

### numeric
- string2number: func get_numeric
- concat 


### wide processing

对数据按照特定格式分割，对每个特征hash和crossing后构造稀疏矩阵。没有使用dense matrix是因为矩阵过大内存放不下。在训练时使用sparse embedding look up 模仿`$wx+b$`的形式解决`tf.dense` 不接受稀疏矩阵输入的问题。

- split by “##”
- hash the split value
- change it to sparse use `tf.SparseTensor`
- cross pv and clks
- concat sparse features use `tf.sparse_concat`

在concat遇到的问题是，每个特征hash后的value 都是[0，bucket)，在concat时需要把特征整理到[0,bucket1+bucket2+bucket3),所以，每个特征hash的值需要
```
feature1 hash： hash value + 0
feautre2 hash：hash value + feature1 bucket
...
```

## build graph

### deep

deep.py

```
dense(1024)
relu
dropout(0.9)

dense(512)
relu
dropout(0.9)

dense(256)
relu
dropout(0.9)

```

### wide

使用`tf.nn.embedding_lookup_sparse`模仿`$wx+b$`的形式解决`tf.dense`不接受稀疏矩阵输入的问题。


```
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
```


## freezing  and load graph

freezing graph 主要是用于固化模型和权重用对跨设备部署

### freezing

freeze.py

```
#!/bin/env python

from tensorflow.python.tools import freeze_graph


prefix = "/data/new/wandd/wd_lab/model_test/"
input_graph_path = prefix + "widendeep.pbtxt"  # 图的pbtxt文件
input_saver_def_path = ""
input_binary = False
output_node_names = "prediction"   # 输出op的名字
restore_op_name = "save"   
filename_tensor_name = "save/Const:0"  #Const:0 是固定格式
clear_devices = True # 是否清楚设备的信息
input_meta_graph = prefix + "my-model.meta" #模型meta
checkpoint_path = prefix + "my-model-300"   #checkpoint
output_graph_filename= "./outGraph.pb"      #输出pb
freeze_graph.freeze_graph(
    input_graph_path, input_saver_def_path, input_binary, checkpoint_path,
    output_node_names, restore_op_name, filename_tensor_name,
    output_graph_filename, clear_devices, "")

```

### load graph

加载freeze好的graph，并且制定 output 和 input op 就可以做预测或者训练。

loadGraph.py

参考：

```
https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
```

## TODO

有几个问题：
- 读取数据不是shuffle
- 读取数据不是多线程效率低
- train.py 可以吧processing 拆出来
- 预加载训练好的模型再训练时候，tensorboard有问题，不知道怎么获取上一次训练的step存到summary里。
- 没写parsing

# wide and deep multigpu

BUG: embedding 的的梯度没有更新,待查.
