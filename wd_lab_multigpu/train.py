#encoding=utf-8
import argparse

import tensorflow as tf
import sys, os, time
import numpy as np
from datetime import datetime
from utils.Parser import ConfParser
from utils.process import multiEmbedding
from utils.model import tower_loss, average_gradients

def deep_train():
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        print("deep_train ")
        print(tf.get_variable_scope())
        # parser

        confParser = ConfParser(conf_path=FLAGS.conf_path)


        with tf.device("/cpu:0"):
            # embedding

              ## TODO conf parse it!
            embedding_para = [(10000, 32, "string", "creativeid"), (250000, 64, "string", "sessionid"),
                              (100000, 32, "string", "userid")]  # TODO
              ##

            emb_list = multiEmbedding(embedding_para)
            emb_feature = [x[3] for x in embedding_para]
            emb_dict = dict(zip(emb_feature, emb_list))

            # iteration
            base_dataset = confParser.dataset(emb_dict=emb_dict, filenames=FLAGS.train_dir)
            batch_size = 2
            batched_dataset = base_dataset.batch(batch_size)
            iterator = batched_dataset.make_initializable_iterator()
            next_element = iterator.get_next()

            ## Create an optimizer that performs gradient descent.



            # Create a variable to count the number of train() calls. This equals the
            # number of batches processed * FLAGS.num_gpus.
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
            # Calculate the learning rate schedule.
            num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                     batch_size)
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                            global_step,
                                            decay_steps,
                                            LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            deep_opt = tf.train.AdagradOptimizer(lr)
            # Calculate the gradients for each model tower.
            tower_grads = []

        for i in range(FLAGS.num_gpus):
            print(i)
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ("deep", i)) as scope:
                    print("deep_train gpu")
                    # Dequeues one batch for the GPU
                    deep_batch, label_batch = iterator.get_next()
                    print("deep_batch")
                    print(deep_batch)
                    print("label_batch")
                    print(label_batch)

                    # Calculate the loss for one tower of the CIFAR model. This function
                    # constructs the entire CIFAR model but shares the variables across
                    # all towers.
                    # Reuse variables for the next tower.
                    # tf.get_variable_scope().reuse_variables()

                    loss = tower_loss(scope, deep_batch, label_batch, batch_size)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = deep_opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)
    # We must calculate the mean of each gradient. Note that this is the

    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = deep_opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)
    sess.run(iterator.initializer)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.model_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
            num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), step, loss_value,
                            examples_per_sec, sec_per_batch))

        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.model_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


if __name__ == "__main__":
    # Constants describing the training process.
    MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100000

    parser = argparse.ArgumentParser()

    # # Basic model parameters.
    # parser.add_argument('--batch_size', type=int, default=2,
    #                     help='Number of images to process in a batch.')

    # parser.add_argument('--data_dir', type=str, default='./',
    #                     help='Path to the CIFAR-10 data directory.')

    # parser.add_argument('--use_fp16', type=bool, default=False,
    #                     help='Train the model using fp16.')

    # parser.add_argument('--train_dir', type=str, default='./',
    #                     help='Directory where to write event logs and checkpoint.')

    # parser.add_argument('--max_steps', type=int, default=30,
    #                     help='Number of batches to run.')

    # parser.add_argument('--num_gpus', type=int, default=2,
    #                     help='How many GPUs to use.')

    # parser.add_argument('--log_device_placement', type=bool, default=False,
    #                     help='Whether to log device placement.')

    # FLAGS = parser.parse_args()
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('batch_size', 2,
                                """Number of datas to process in a batch.""")
    tf.app.flags.DEFINE_integer('max_steps', 47,
                                """Number of datas to process in a batch.""")
    tf.app.flags.DEFINE_integer('num_gpus', 2,
                                """Number of datas to process in a batch.""")

    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Number of datas to process in a batch.""")

    tf.app.flags.DEFINE_string('model_dir','../demo_data/',"""train dir""")
    tf.app.flags.DEFINE_string('train_dir','../demo_data/tiny_tr/part-00000',"""train dir""")
    tf.app.flags.DEFINE_string('conf_path','../demo_data/features.yaml',"""feature.yaml path""")


    ## load config yaml
    #yaml_config = yaml.load(open("/data/wandd/demo_data/features.yaml"))
    #COLUMNS = yaml_config['using_features_dl']
    #FEATURE_CONF = yaml_config['features_conf']
    #CARTESIAN_CROSS = yaml_config['cartesian_cross']
    #LABEL_COLUMN = "label"
    #POSTERIOR_PREFIX = ["poPV", "poCLK", "poCTR", "poCOEC_st", "poCOEC_em"]
    #CARTESIAN_CROSS_DICT = {}


    # If a model is trained with multiple GPUs, prefix all Op names with tower_name
    # to differentiate the operations. Note that this prefix is removed from the
    # names of the summaries when visualizing a model.
    deep_train()

    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
    #                                         log_device_placement=True)) as sess:

#     sess.run(tf.global_variables_initializer())

#     #sess.run(iterator.initializer)
#     deep_train()