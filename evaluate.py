""" Based on point net evaluation process
downladed from: https://github.com/charlesq34/pointnet
"""

import tensorflow as tf
import numpy as np
import math
import argparse
import socket
import os
import sys
from pyhocon import ConfigFactory

from pointcloud_conv_net import Network

import provider


BASE_DIR = '../../../../../'
pv = provider.ClassificationProvider(False)
pv.BASE_DIR = BASE_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 1]')
parser.add_argument('--model_path', default='epoch_250/model.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--config', type=str, default='pointconv.conf',
                    help='Config to use [default: pointconv]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
conf = ConfigFactory.parse_file('{0}'.format(FLAGS.config))
NUM_POINT = conf.get_list('network.pool_sizes_sigma')[0][0]

DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
               open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))]

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TEST_FILES = pv.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)





def evaluate(num_votes):
    is_training = False

    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
        labels_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
        is_training_pl = tf.placeholder(tf.bool, shape=())
        is_evaluate_pl = tf.placeholder(tf.bool, shape=())

        # Get model and loss
        # pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
        network = Network(conf.get_config('network'))
        pred = network.build_network(pointclouds_pl, is_training_pl,is_evaluate_pl)
        loss = network.get_loss(pred, labels_pl)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'is_evaluate_pl': is_evaluate_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops, num_votes)


def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    is_evaluate = True
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '----')
        current_data, current_label = pv.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:, 0:conf.get_int('no_points_sample'), :]
        current_label = np.squeeze(current_label)
        print(current_data.shape)

        file_size = current_data.shape[0]
        num_batches = int(math.ceil(file_size * 1.0 / BATCH_SIZE))
        print(file_size)
        print(num_batches)

        for batch_idx in range(num_batches):

            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            print('start_idx : {0}'.format(start_idx))
            cur_batch_size = current_data[start_idx:end_idx].shape[0]

            # Aggregating BEG
            batch_loss_sum = 0  # sum of losses for the batch
            batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES))  # score for classes
            batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES))  # 0/1 for classes
            for vote_idx in range(num_votes):

                if (vote_idx == num_votes - 1):
                    test_data = current_data[start_idx:end_idx, :NUM_POINT, :]
                else:
                    test_data = current_data[start_idx:end_idx,
                                   np.random.choice(np.arange(current_data.shape[1]), NUM_POINT, replace=False), :]

                if (cur_batch_size < BATCH_SIZE):
                    test_data = np.concatenate([test_data, np.zeros((BATCH_SIZE - cur_batch_size, NUM_POINT, 3))],
                                                  0)
                    current_label_feed = np.concatenate(
                        [current_label[start_idx:end_idx], np.ones((BATCH_SIZE - cur_batch_size))],
                        0)
                else:
                    current_label_feed = current_label[start_idx:end_idx]

                test_data =  pv.rotate_point_cloud_by_angle(test_data,vote_idx/float(num_votes) * np.pi * 2)

                feed_dict = {ops['pointclouds_pl']: test_data,
                             ops['labels_pl']: current_label_feed,
                             ops['is_training_pl']: is_training,
                             ops['is_evaluate_pl']: is_evaluate}
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                              feed_dict=feed_dict)
                batch_pred_sum += pred_val[0:cur_batch_size]
                batch_pred_val = np.argmax(pred_val[0:cur_batch_size], 1)
                for el_idx in range(cur_batch_size):
                    batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
                batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))

            pred_val = np.argmax(batch_pred_sum, 1)


            correct = np.sum(pred_val == current_label[start_idx:end_idx])

            total_correct += correct
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum

            for i in range(start_idx, min(end_idx, file_size)):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx] == l)
                fout.write('%d, %d\n' % (pred_val[i - start_idx], l))

    log_string('total seen : {0}'.format(total_seen))
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
    np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))


if __name__ == '__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=10)
    LOG_FOUT.close()
