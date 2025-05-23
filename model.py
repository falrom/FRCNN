import tensorflow as tf
import numpy as np
import os
import time

from layers import DRRN
from layers import MSE_float
from layers import loss_L1
from layers import PSNR_float
from layers import PSNR_uint8
from layers import Outputs
from layers import dataset_TFR
from layers import dataset_YUV

from utils import print_info
from utils import record_info
from utils import get_time_str
from utils import YUVread
from utils import YUVwrite


class MyEstimator:

    def __init__(self, count_B, count_U, channel, QP, video_name,
                 use_BN_at_begin=True, use_BN_in_ru=True, use_BN_at_end=True):

        # Run flags:
        self.MODE_TRAIN = 'TRAIN'
        self.MODE_EVAL = 'EVAL'
        self.MODE_PREDICT = 'PREDICT'
        self.info_top = {}

        # Hyper-parameters:
        self.count_B = count_B
        self.count_U = count_U
        self.channel = channel
        self.use_BN_at_begin = use_BN_at_begin
        self.use_BN_in_ru = use_BN_in_ru
        self.use_BN_at_end = use_BN_at_end
        self.net_name = 'DRRNoverfit' + '_B' + str(self.count_B) + 'U' + str(self.count_U) + 'C' + str(self.channel)
        self.info_top['net_name'] = self.net_name
        self.info_top['use_BN_at_begin'] = str(self.use_BN_at_begin)
        self.info_top['use_BN_in_ru'] = str(self.use_BN_in_ru)
        self.info_top['use_BN_at_end'] = str(self.use_BN_at_end)

        # Video information:
        self.QP = QP
        self.video_name = video_name
        self.info_top['video_name'] = self.video_name
        self.info_top['QP'] = str(self.QP)

        # Others:
        self.use_L1_loss = False

        print('\n\n********** MyEstimator **********')
        print_info([self.info_top])
        print('********** *********** **********')

    def def_model(self, mode, height, width):
        # inputs:
        with tf.name_scope(name='inputs'):
            self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1], name='inputs')
        with tf.name_scope(name='labels'):
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1], name='labels')

        # state:
        with tf.variable_scope('states'):
            self.is_training = tf.placeholder(dtype=tf.bool, name='train_flag')
            if mode == self.MODE_TRAIN:
                self.global_step = tf.Variable(0, trainable=False, name=tf.GraphKeys.GLOBAL_STEP)
                tf.add_to_collections([tf.GraphKeys.GLOBAL_STEP, 'var_list'], self.global_step)
                self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        # inference:
        self.drrn_predictions = DRRN(
            x=self.inputs,
            count_B=self.count_B,
            count_U=self.count_U,
            out_depth=self.channel,
            is_training=self.is_training,
            name=self.net_name,
            use_BN_at_begin=self.use_BN_at_begin,
            use_BN_in_ru=self.use_BN_in_ru,
            use_BN_at_end=self.use_BN_at_end,
            collections=['var_list']
        )

        # tails:
        if mode == self.MODE_TRAIN:
            self.mse = MSE_float(predictions=self.drrn_predictions, labels=self.labels)
            self.psnr_float = PSNR_float(mse=self.mse)
            self.loss = self.mse
            if self.use_L1_loss:
                self.loss_L1 = loss_L1(self.drrn_predictions, labels=self.labels)
                self.loss = self.loss_L1
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.optimizer == 'sgd':
                    self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                        self.loss, global_step=self.global_step)
                elif self.optimizer == 'adam':
                    self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                        self.loss, global_step=self.global_step)
                else:
                    raise Exception('ERROR: Unknown optimizer name.')
            self.variable_init = tf.global_variables_initializer()
        if mode == self.MODE_EVAL:
            self.mse = MSE_float(predictions=self.drrn_predictions, labels=self.labels)
            self.psnr_float = PSNR_float(mse=self.mse)
            self.outputs = Outputs(predictions=self.drrn_predictions)
            self.psnr_uint8 = PSNR_uint8(outputs=self.outputs, labels=self.labels)
        if mode == self.MODE_PREDICT:
            self.outputs = Outputs(predictions=self.drrn_predictions)

    def train(self,
              time_str=None,
              height=41, width=41,
              train_batch_size=200,
              test_batch_size=500,
              steps=None, max_steps=None,
              test_interval=50,
              save_interval=500,
              learning_rate=0.001,
              decay=None,
              use_L1_loss=False,
              optimizer='adam'):

        # params:
        time_str = time_str or get_time_str()
        # self.learning_rate = learning_rate
        self.use_L1_loss = use_L1_loss
        self.optimizer = optimizer.lower()
        save_max_psnr = [30.0]

        # paths:
        log_dir = os.path.join('./logs', self.net_name, self.video_name + '_QP' + str(self.QP), time_str)
        backup_dir = os.path.join('./checkpoints', self.net_name, self.video_name + '_QP' + str(self.QP), time_str)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        latest_ckpt_path = tf.train.latest_checkpoint(backup_dir)
        train_data_path = './data/TFRdata/' + \
                          self.video_name + '_QP' + str(self.QP) + '_IP_' + str(height) + 'x' + str(width) + \
                          '.tfrecords'
        test_data_path = train_data_path

        # info:
        self.info_train = {}
        if latest_ckpt_path:
            self.info_train['old_ckpt'] = latest_ckpt_path
        self.info_train['time_str'] = time_str
        self.info_train['learning_rate'] = str(learning_rate)
        if decay:
            self.info_train['lr_decay'] = str(decay)
        self.info_train['patch_size'] = str(height) + 'x' + str(width)
        self.info_train['train_batch_size'] = str(train_batch_size)
        self.info_train['test_batch_size'] = str(test_batch_size)
        self.info_train['log_dir'] = log_dir
        self.info_train['backup_dir'] = backup_dir
        self.info_train['train_data'] = train_data_path
        self.info_train['test_data'] = test_data_path
        self.info_train['loss_function'] = 'L2-MSE'
        if use_L1_loss:
            self.info_train['loss_function'] = 'L1-absolute_difference'
        self.info_train['optimizer'] = optimizer
        print('\n\n********** Train **********')
        print_info([self.info_train])
        print('********** ***** **********')
        record_info([self.info_top, self.info_train], os.path.join(backup_dir, 'info.txt'))
        record_info([self.info_top, self.info_train], os.path.join(log_dir, 'info.txt'))

        # define graph:
        print('\n** Define graph...')
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            # define model:
            self.def_model(self.MODE_TRAIN, height, width)

            # logs:
            log_vars = tf.summary.merge_all()
            log_train_MSE = tf.summary.scalar('MSE_train', self.mse)
            log_test_MSE = tf.summary.scalar('MSE_test', self.mse)
            log_train_PSNR = tf.summary.scalar('PSNR_train', self.psnr_float)
            log_test_PSNR = tf.summary.scalar('PSNR_test', self.psnr_float)
            log_lr = tf.summary.scalar('learning_rate', self.learning_rate)
            log_writer = tf.summary.FileWriter(log_dir)
            log_writer.add_graph(self.train_graph)
            log_writer.flush()

            # savers:
            saver_var = tf.train.Saver(tf.get_collection('var_list'), name='saver_var')
            saver_all = tf.train.Saver(max_to_keep=0, name='saver_all')
        print('Done.')

        # datasets:
        print('\n** Generate datasets...')
        print('train data path:', train_data_path)
        print('test  data path:', test_data_path)
        with self.train_graph.as_default():
            get_train_batch = dataset_TFR(train_data_path, train_batch_size, height, width, 76850)
            get_test_batch = dataset_TFR(test_data_path, test_batch_size, height, width, 76850)
        print('Done.')

        print('\n** Initialize and prepare...')

        # init:
        sess = tf.Session(graph=self.train_graph)
        if latest_ckpt_path:
            saver_all.restore(sess, latest_ckpt_path)
        else:
            sess.run(self.variable_init)
        step = tf.train.global_step(sess, self.global_step)
        steps_to_run = None
        if steps or max_steps:
            steps_to_run = steps or max(max_steps - step, 0)

        # define process functions:
        def train_once(step, pring_log=True):
            train_batch = sess.run(get_train_batch)
            lr = learning_rate
            if decay:
                lr = learning_rate * (decay ** step)
            feed_dic = {
                self.inputs: train_batch['input'],
                self.labels: train_batch['label'],
                self.is_training: True,
                self.learning_rate: lr}
            mse, mse_log, psnr, psnr_log, lr_log, _ = sess.run(
                [self.mse, log_train_MSE, self.psnr_float, log_train_PSNR, log_lr, self.train_op], feed_dic)
            log_writer.add_summary(mse_log, step)
            log_writer.add_summary(psnr_log, step)
            log_writer.add_summary(lr_log, step)
            if pring_log:
                print('step: %d  lr: %.8f  train-loss: %.10f  train-PSNR: %.6f' % (step, lr, mse, psnr))

        def test_once(step, pring_log=True):
            test_batch = sess.run(get_test_batch)
            feed_dic = {
                self.inputs: test_batch['input'],
                self.labels: test_batch['label'],
                self.is_training: False}
            mse, mse_log, psnr, psnr_log = sess.run(
                [self.mse, log_test_MSE, self.psnr_float, log_test_PSNR], feed_dic)
            log_writer.add_summary(mse_log, step)
            log_writer.add_summary(psnr_log, step)
            log_writer.flush()
            if pring_log:
                print('--------------------------------------------------------------')
                print('step: %d  test-loss: %.10f  test-PSNR: %.6f' % (step, mse, psnr))
                print('--------------------------------------------------------------')
            return psnr

        def save_once(step, pring_log=True):
            save_path = os.path.join(backup_dir, get_time_str())
            saver_all.save(
                sess=sess,
                save_path=save_path,
                global_step=step,
                write_meta_graph=False)
            if pring_log:
                print('save:', save_path)
            return save_path

        print('Done.')

        # run:
        save_path = None
        print('\n** Begin training:')
        if latest_ckpt_path is None:
            test_once(0)
            save_path = save_once(0)
        else:
            test_once(step)

        t = time.time()
        save_flag_max = False
        save_flag_final = False
        while (steps_to_run is None) or (steps_to_run > 0):
            save_flag_final = True
            step = tf.train.global_step(sess, self.global_step) + 1
            train_once(step)
            if (step % test_interval) == 0:
                print('time: train_%d %.6fs' % (test_interval, time.time() - t))
                t = time.time()
                tmp = test_once(step)
                print('time: test_once %.6fs' % (time.time() - t))
                if tmp > np.mean(save_max_psnr[-5:]):
                    save_max_psnr.append(tmp)
                    save_flag_max = True
                t = time.time()
            if ((step % save_interval) == 0) or save_flag_max:
                t = time.time()
                save_path = save_once(step)
                if save_flag_max:
                    print(save_max_psnr[-6:])
                save_flag_max = False
                save_flag_final = False
                print('time: save_once %.6fs' % (time.time() - t))
                t = time.time()
            if steps_to_run is not None:
                steps_to_run -= 1

        if save_flag_final:
            save_path = save_once(step)
        sess.close()
        print('\nALL DONE.')
        return save_path

    def evaluate(self,
                 ckpt_path,
                 height=1080,
                 width=1920,
                 need_output=False):

        # paths:
        input_path = './data/videos/' + self.video_name + '_QP' + str(self.QP) + '_IP_rec.yuv'
        label_path = './data/videos/' + self.video_name + '.yuv'
        output_path = None
        if need_output:
            output_path = './data/videos/' + self.video_name + '_QP' + str(self.QP) + '_IP_rec_' + \
                          'B' + str(self.count_B) + 'U' + str(self.count_U) + 'C' + str(self.channel) + '.yuv'

        # info:
        self.info_evaluate = {}
        self.info_evaluate['image_size'] = str(height) + 'x' + str(width)
        self.info_evaluate['input_data'] = input_path
        self.info_evaluate['label_data'] = label_path
        self.info_evaluate['checkpoint'] = ckpt_path
        if need_output:
            self.info_evaluate['output_path'] = output_path
        print('\n\n********** evaluate **********')
        print_info([self.info_evaluate])
        print('********** ******** **********')

        # define graph:
        print('\n** Define graph...')
        self.evaluate_graph = tf.Graph()
        with self.evaluate_graph.as_default():
            self.def_model(self.MODE_EVAL, height, width)
            saver_all = tf.train.Saver(max_to_keep=0, name='saver_all')
        print('Done.')

        # datasets:
        print('\n** Generate datasets...')
        print('input data path:', input_path)
        print('label data path:', label_path)
        with self.evaluate_graph.as_default():
            get_batch = dataset_YUV(input_path=input_path, label_path=label_path, batch_size=1, patch_height=height,
                                    patch_width=width, shuffle=False, repeat=1)
        print('Done.')

        # init:
        print('\n** Initialize and prepare...')
        sess = tf.Session(graph=self.evaluate_graph)
        saver_all.restore(sess, ckpt_path)
        print('Done.')

        # run:
        print('\n** Evaluating:')
        t = time.time()
        psnr_uint8_a_collection = []
        psnr_uint8_b_collection = []
        psnr_gain_collection = []
        outputs = None
        for frame, batch in enumerate(get_batch):
            # run reference:
            feed_dic = {
                self.drrn_predictions: batch['input'],
                self.labels: batch['label'],
                self.is_training: False}
            mse_b, psnr_float_b, psnr_uint8_b = sess.run([self.mse, self.psnr_float, self.psnr_uint8], feed_dic)
            print('Frame %d before process: MSE = %.10f, PSNR(float) = %.6f, PSNR = %.6f' % \
                  (frame, mse_b, psnr_float_b, psnr_uint8_b), flush=True)
            # run results:
            feed_dic = {
                self.inputs: batch['input'],
                self.labels: batch['label'],
                self.is_training: False}
            if need_output:
                mse_a, psnr_float_a, psnr_uint8_a, batch_outputs = sess.run(
                    [self.mse, self.psnr_float, self.psnr_uint8, self.outputs], feed_dic)
                if outputs is None:
                    outputs = batch_outputs
                else:
                    outputs = np.concatenate([outputs, batch_outputs])
            else:
                mse_a, psnr_float_a, psnr_uint8_a = sess.run([self.mse, self.psnr_float, self.psnr_uint8], feed_dic)
            print('Frame %d after  process: MSE = %.10f, PSNR(float) = %.6f, PSNR = %.6f' % \
                  (frame, mse_a, psnr_float_a, psnr_uint8_a), flush=True)
            psnr_gain = psnr_uint8_a - psnr_uint8_b
            psnr_uint8_a_collection.append(psnr_uint8_a)
            psnr_uint8_b_collection.append(psnr_uint8_b)
            psnr_gain_collection.append(psnr_gain)
            print('PSNR Gain: %.6f' % psnr_gain)
            print('--------------------------------------------------------------------------------------')

        psnr_uint8_a_mean = (sum(psnr_uint8_a_collection) / len(psnr_uint8_a_collection))
        psnr_uint8_b_mean = (sum(psnr_uint8_b_collection) / len(psnr_uint8_b_collection))
        psnr_gain_mean = (sum(psnr_gain_collection) / len(psnr_gain_collection))

        print('Average PSNR before process: %.6f' % psnr_uint8_b_mean)
        print('Average PSNR after  process: %.6f' % psnr_uint8_a_mean)
        print('Average PSNR gain: %.6f' % psnr_gain_mean)
        print('Time cost: %.6fs' % (time.time() - t))
        print('Done.')

        # write results:
        if need_output:
            print('\n** Write results...')
            _, U, V = YUVread(input_path, [height, width])
            outputs = np.reshape(outputs, [-1, height, width])
            YUVwrite(outputs, U, V, output_path)
            print('Output path:', output_path)
            print('Done.\n')

        return psnr_gain_mean

    def predict(self):
        pass
