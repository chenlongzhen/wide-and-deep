import tensorflow as tf
import argparse 
#from reader import *

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return  graph


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="./outGraph.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
        
    # We access the input and output nodes 
    emb_placeholder = graph.get_tensor_by_name('prefix/INPUT/emb_placeholder:0')
    onehot_placeholder = graph.get_tensor_by_name('prefix/INPUT/onehot_placeholder:0')
    wide_placeholder = graph.get_tensor_by_name('prefix/INPUT/wide_placeholder:0')

    prediction = graph.get_tensor_by_name("prefix/prediction:0")

#    COLUMNS = ['label', 'browser', 'city', 'creativeid', 'iid_clked', 'iid_imp', 'itemtype', 'network', 'operation', 'os', 'province', 'psid_abs', 'pvday', 'pvhour', 'read', 'sessionid', 'source', 'userid']
#    numeric_col = None
#    emb_col = ['creativeid', 'sessionid', 'userid']
#    onehot_col = ['psid_abs', 'operation', 'itemtype', 'pvday', 'pvhour', 'network', 'read', 'browser', 'source', 'os', 'province']
#    wide_col = ['iid_imp', 'iid_clked']
#    data_reader = reader("/data/new/dis_with_wide/train", "/data/new/dis_with_wide/test", COLUMNS, numeric_col, 3)
#
#    emb_data, onehot_data, wide_data, label_data = data_reader.get_test_batch(emb_col, onehot_col, numeric_col, wide_col)

#    print("emb_data")
#    print(emb_data)
#
#    
#    print("onehot_data")
#    print(onehot_data)
#
#  
#    print("wide_data")
#    print(wide_data)

    # test data !
    emb_data = [['missing','C2D1F06A-1826-44E3-AFBE-6B85E3A9D971','0\t'],
 ['1-904953','8C3B7DB7-9C99-DC7D-114C-FC43C9294F83','0\t'],
 ['11-1620558','missing','0\t']]

    onehot_data = [['2','2','12','missing','14','unknown','0','missing','missing','missing'
,'120000'],
 ['1','3','1','missing','07','cellular','0','missing','missing','missing'
,'130000'],
 ['3','2','11','missing','10','wifi','0','missing','missing','missing'
 ,'360000']] 

    wide_data = [['missing','12_2031476##5_65147731##1_899820##1_902763##12_2031199##5_65004871##5_64880165##11_1545324##5_65136302##5_64829190##5_65336557##5_65429910##5_64723538##5_64853867##12_2031941##1_905472##12_2030950##5_64819456##5_64773032##12_2031849##12_2031774##5_64719726##5_64688473##5_65332112##5_65168429##33_65175651##5_65495318##5_65272946##5_65113228##5_64782371##5_65489857##5_65499582##5_65518950##5_65495318##5_64865010##5_65332112##5_65462709##6_3957525##35_222##12_2030379##5_65236374##5_65261581##1_905443##1_904843##5_65502920##5_65464645##5_65457078##1_905160##5_65429686##5_64971544##5_65242931##5_64766423##5_65417741##5_65221856##5_65509317##5_64811067##5_65500672##5_65488235##5_64833298##12_2030870##5_65205918##5_64765918##5_65437290##12_2032116##5_65449395##12_2031345##5_65237073##5_65277928##12_2028884##6_3765877##5_65277928##5_64721482##1_904840##5_65149972##5_65486779##5_65520123##5_65401212##5_64822630##6_3970267##5_65247774##5_64802208##5_65524499##5_64765918##5_64768868##5_64989172##5_65341076##5_64984180##5_65423326##5_65524423##5_65491871##6_3952650##5_65300742##5_65346780##6_3644801##5_64773624##5_65227209##1_905145##5_65481011##5_65106044##5_65337328##5_65481011##5_65146783'],['1-904953','5_65108196##5_64768435##12_2030508##5_65206165##5_64926008##5_64681180##5_64895830##5_65377516##5_64847911##14_2029793##6_3957182##5_65417741##5_65063826##5_65108639##1_905145##12_2028520##5_64969493##5_65275610##5_64944003##5_65261581##5_65109341##1_904757##5_65476407##5_65012113##1_905312##6_3970267##1_904807##6_3765877##1_904843##1_904758##5_65489857##1_904479##1_904844##12_2032232##16_53720##5_65518950##11_1646734##5_65084368##5_65197000##5_65491871##5_65242931##1_904913##5_65367363##5_65249021##5_65437290##5_65149972##5_65139621##5_65462709##5_65537258##5_65481011##5_65403192##5_65505123##5_65236374##5_65536237##5_65491871##12_2029583##5_65056783##5_65470261##5_65465943##5_65428776##3_119238##5_65520123##5_65453450##5_65449395##5_65455501##5_65426567##5_64887697##5_65452979##14_2031124##5_64765840##5_65425781##5_65250070##1_905585##16_54391##3_119373##5_65532591##14_2032143##5_65179211##12_2029552##5_65565187##5_65567031##5_65570173##12_2034481##1_905608##5_65156290##6_3963417##5_64903870##1_903963##5_65524001##5_65574352##5_65346780##5_64926008##12_2030950##5_64902989##5_65063826##16_53634##5_65107220##5_65143873##12_2032573##5_64997818##5_65499582##1_905334##11_1623509##5_64935623##33_64878763##5_65518950'],['missing','missing']]  

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we didn't initialize/restore anything, everything is stored in the graph_def
        prediction = sess.run(prediction, feed_dict={
        emb_placeholder:emb_data,
        onehot_placeholder:onehot_data,
        wide_placeholder:wide_data
        })
        print(prediction) # [[ False ]] Yay, it works!

    
   #https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
   #prefix/INPUT/emb_placeholder
   #prefix/INPUT/onehot_placeholder
   #prefix/INPUT/wide_placeholder
