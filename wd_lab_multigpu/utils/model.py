#encoding=utf-8
import tensorflow as tf
import re


def nn_layer(data_input,in_dim, out_dim, name,use_relu = True, use_drop = True):
    """
    """
    with tf.variable_scope(name) as scope:
        weights = _variable_with_weight_decay('weights',
                                         shape=[in_dim, out_dim],
                                         stddev=5e-2,
                                         wd=0.0)

        biases = _variable_on_cpu('biases', [out_dim], tf.constant_initializer(0.1))
        activation = tf.add(tf.matmul(data_input,weights), biases, name = scope.name)

    if use_relu:
            activation = tf.nn.relu(activation, name = 'relu')

    if use_drop:
            activation = tf.nn.dropout(activation, keep_prob = 0.9, name = 'drop')

    return activation



def deep_net_new(deep_input, batch_size):
    """
       deep net new
    """
    layer1 = nn_layer(deep_input, deep_input.shape[1], 1024, name = 'dense1')

    layer2 = nn_layer(layer1, 1024, 512, name = 'dense2')

    layer3 = nn_layer(layer2, 512, 256, name = 'dense3')

    layer4 = nn_layer(layer3, 256, 1, use_relu = False, use_drop = False, name = 'deep_logit')

    return layer4

#def  _activation_summary(x):
#    """Helper to create summaries for activations.
#    Creates a summary that provides a histogram of activations.
#    Creates a summary that measures the sparsity of activations.
#    Args:
#      x: Tensor
#    Returns:
#      nothing
#    """
#    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
#    # session. This helps the clarity of presentation on tensorboard.
#    #tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
#    tf.summary.histogram(tensor_name + '/activations', x)
#    tf.summary.scalar(tensor_name + '/sparsity',
#                                         tf.nn.zero_fraction(x))

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
    return var

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

# 梯度求平均
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
    return average_grads


# get all loss （ this tower ）

def tower_loss(scope, deep_lines, labels, batch_size):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].
    Returns:
     Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    # logits = deep_net(deep_input=deep_lines, mode = "train")
    logits = deep_net_new(deep_input=deep_lines, batch_size=batch_size)
    print("logits")
    print(logits)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    # _ = cifar10.loss(logits, labels)
    labels = tf.reshape(labels, [-1, 1])
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)
    tf.add_to_collection('losses', loss)  # ！加入集合操作

    print("loss")
    print(loss)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')  # !!! add all losses ?

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % "deep", '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss