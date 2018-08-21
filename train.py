""" Based on point net training process
downladed from: https://github.com/charlesq34/pointnet
"""

import tensorflow as tf
import socket
import math
import os
import sys
import numpy as np
import datetime
from pyhocon import ConfigFactory
import argparse

np.random.seed(0)
from pointcloud_conv_net import Network
from provider import ClassificationProvider


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():

    TRAIN_FILES = provider.getTrainDataFiles()
    TEST_FILES = provider.getTestDataFiles()

    with tf.Graph().as_default():
        tf.set_random_seed(0)
        with tf.device('/gpu:' + str(GPU_INDEX)):


            pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE,NUM_POINT, 3))
            labels_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
            is_training_pl = tf.placeholder(tf.bool, shape=())
            is_evaluate_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            #pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            network = Network(conf.get_config('network'))
            pred = network.build_network(pointclouds_pl,is_training_pl,is_evaluate_pl,bn_decay)
            loss = network.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                MOMENTUM = 0.9
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'is_evaluate_pl': is_evaluate_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        global total_paramers
        total_paramers = 0
        for varaible in tf.trainable_variables():
            shape = varaible.get_shape()
            var_parameter = 1
            for dim in shape:
                var_parameter *= dim.value
            print ("variabile : {0} , {1}".format(varaible.name,var_parameter))
            log_string("variabile : {0} , {1}".format(varaible.name,var_parameter))
            total_paramers += var_parameter

        log_string("total parameters : {0}".format(total_paramers))

        def train_one_epoch(sess, ops, train_writer,epoch):
            global total_paramers
            """ ops: dict mapping from string to tf ops """
            is_training = True
            is_evaluate = False

            # Shuffle train files
            train_file_idxs = np.arange(0, len(TRAIN_FILES))
            np.random.shuffle(train_file_idxs)


            for fn in range(len(TRAIN_FILES)):
                log_string('----' + str(fn) + '-----')
                current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])

                current_data = current_data[:, : , :]
                current_data, current_label, idx = provider.shuffle_data(current_data, np.squeeze(current_label))

                current_label = np.squeeze(current_label)

                file_size = current_data.shape[0]
                num_batches = file_size // BATCH_SIZE

                total_correct = 0
                total_seen = 0
                loss_sum = 0

                total_correct_last = 0.
                total_seen_last = 0.
                for batch_idx in range(num_batches):
                    #print ("batch_idx : {0}".format(batch_idx))
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = (batch_idx + 1) * BATCH_SIZE

                    if (conf.get_bool('network.with_rotations')):
                        rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, np.random.choice(
                            np.arange(conf.get_int('no_points_sample')), NUM_POINT, False), :])
                        augmented_data = provider.translate_point_cloud(rotated_data)

                    else:
                        augmented_data = provider.translate_point_cloud(current_data[start_idx:end_idx, np.random.choice(np.arange(conf.get_int('no_points_sample')),NUM_POINT,False), :])

                    feed_dict = {ops['pointclouds_pl']: augmented_data,
                                 ops['labels_pl']: current_label[start_idx:end_idx],
                                 ops['is_training_pl']: is_training,
                                 ops['is_evaluate_pl']: is_evaluate}

                    summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                                     ops['train_op'], ops['loss'], ops['pred']],
                                                                    feed_dict=feed_dict)

                    train_writer.add_summary(summary, step)
                    pred_val = np.argmax(pred_val, 1)
                    correct = np.sum(pred_val == current_label[start_idx:end_idx])

                    total_correct += correct
                    total_seen += BATCH_SIZE

                    if (batch_idx % 10 == 1):
                        print ('accuracy : {0}'.format(total_correct_last / total_seen_last))
                        total_correct_last = 0.0
                        total_seen_last = 0.0

                    total_correct_last += correct
                    total_seen_last += BATCH_SIZE

                    loss_sum += loss_val


                log_string('mean loss: %f' % (loss_sum / float(num_batches)))
                log_string('accuracy: %f' % (total_correct / float(total_seen)))
            return step

        def eval_one_epoch(sess, ops, epoch_index):
            global total_paramers

            is_training = False
            is_evaluate = False
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES))

            for fn in range(len(TEST_FILES)):
                log_string('----' + str(fn) + '-----')
                current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
                current_data = current_data[:, 0:NUM_POINT, :]
                current_label = np.squeeze(current_label)

                file_size = current_data.shape[0]
                #num_batches = file_size // BATCH_SIZE
                num_batches = int(math.ceil(file_size * 1.0 / BATCH_SIZE))
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = (batch_idx + 1) * BATCH_SIZE
                    cur_batch_size = current_data[start_idx:end_idx].shape[0]

                    if (cur_batch_size < BATCH_SIZE):
                        current_data_feed = np.concatenate([current_data[start_idx:end_idx, :, :], np.zeros((BATCH_SIZE - cur_batch_size, NUM_POINT, 3))], 0)
                        current_label_feed = np.concatenate(
                            [current_label[start_idx:end_idx], np.ones((BATCH_SIZE - cur_batch_size))],
                            0)
                    else:
                        current_label_feed = current_label[start_idx:end_idx]
                        current_data_feed = current_data[start_idx:end_idx, :, :]

                    feed_dict = {ops['pointclouds_pl']: current_data_feed,
                                 ops['labels_pl']: current_label_feed,
                                 ops['is_training_pl']: is_training,
                                 ops['is_evaluate_pl']: is_evaluate}
                    loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)

                    pred_val = np.argmax(pred_val[:cur_batch_size], 1)
                    correct = np.sum(pred_val == current_label[start_idx:end_idx])
                    total_correct += correct
                    total_seen += cur_batch_size
                    loss_sum += (loss_val * BATCH_SIZE)
                    for i in range(start_idx, min(end_idx,file_size)):
                        l = current_label[i]
                        total_seen_class[l] += 1
                        total_correct_class[l] += (pred_val[i - start_idx] == l)

            log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
            log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

            return np.asscalar(total_correct / float(total_seen))

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer,epoch)
            acc = eval_one_epoch(sess, ops, epoch)

            # Save the variables to disk.
            if (epoch % 50 == 0 ):
                mkdir_ifnotexists(os.path.join(LOG_DIR, 'epoch_{0}'.format(epoch)))
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
                log_string("Model saved in file: %s" % save_path)
                os.system("""cp {0} "{1}" """.format(os.path.join(LOG_DIR, 'model*'), os.path.join(LOG_DIR,'epoch_{0}'.format(epoch))))
                os.system("""cp {0} "{1}" """.format(os.path.join(LOG_DIR, 'checkpoint*'), os.path.join(LOG_DIR,'epoch_{0}'.format(epoch))))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname', type=str, default='pcnn', help='expiriment name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--config', type=str, default='confs/pointconv.conf', help='Config to use [default: pointconv]')
    parser.add_argument('--hypothesis', type=str, default='', help='Document experiment hypothesis')
    FLAGS = parser.parse_args()

    conf = ConfigFactory.parse_file('{0}'.format(FLAGS.config))


    expirment_name = FLAGS.expname
    timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())

    GPU_INDEX = FLAGS.gpu
    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_CLIP = 0.99
    OPTIMIZER = 'adam'
    total_paramers = 0
    MAX_EPOCH = conf.get_int('max_epoch')
    BASE_LEARNING_RATE = conf.get_float('base_learning_rate')
    DECAY_STEP = conf.get_int('decay_step')
    DECAY_RATE = conf.get_float('decay_rate')
    BN_DECAY_DECAY_STEP = float(DECAY_STEP)
    hostname = socket.gethostname().split('.')[0]
    BATCH_SIZE = conf.get_int('batch_size')
    LOG_DIR = os.path.join('./exp_results/{0}/{1}/{2}'.format(expirment_name, hostname, GPU_INDEX), timestamp)

    mkdir_ifnotexists('./exp_results')
    mkdir_ifnotexists('./exp_results/{0}/'.format(expirment_name))
    mkdir_ifnotexists('./exp_results/{0}/{1}'.format(expirment_name,hostname))
    mkdir_ifnotexists('./exp_results/{0}/{1}/{2}'.format(expirment_name, hostname,GPU_INDEX))


    os.mkdir(LOG_DIR)

    mkdir_ifnotexists(os.path.join(LOG_DIR, 'layers'))

    os.system("""cp {0} "{1}" """.format('pointcloud_conv_net.py', LOG_DIR))
    os.system("""cp {0} "{1}" """.format('evaluate.py', LOG_DIR))
    os.system("""cp {0} "{1}" """.format('tf_util.py', LOG_DIR))
    os.system("""cp {0} "{1}" """.format('provider.py', LOG_DIR))
    os.system("""cp train.py "{0}" """.format(LOG_DIR))
    os.system("""cp {0} "{1}" """.format(FLAGS.config,LOG_DIR))
    os.system("""cp ./confs/{0}.conf "{1}" """.format('pointconv', LOG_DIR))
    os.system("""cp ./layers/* "{0}" """.format(os.path.join(LOG_DIR,'layers')))

    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')

    log_string(FLAGS.hypothesis)
    log_string('tf version : {0}'.format(tf.__version__))

    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(GPU_INDEX)

    provider = ClassificationProvider()
    NUM_CLASSES = 40

    NUM_POINT = conf.get_list('network.pool_sizes_sigma')[0][0]
    print ("num point : {0}".format(NUM_POINT))

    log_string("experiment name : {0}".format(expirment_name))
    train()