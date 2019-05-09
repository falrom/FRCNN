import tensorflow as tf
import numpy as np
from utils import calculate_variables
from utils import YUVread

USE_MY_OWN_BN_IMPLEMENTATION = False


def BN_relu_conv(x, w, b,
                 is_training,
                 name,
                 use_BN=True,
                 use_ReLU=True,
                 stride=1,
                 decay=0.99,
                 pad_mode='SAME',
                 summary=False,
                 trainable=True,
                 collections=None):
    """BN layer + ReLU + conv2D."""
    with tf.variable_scope(name_or_scope=name):
        node = x
        # BN:
        if use_BN:
            if not USE_MY_OWN_BN_IMPLEMENTATION:  # Tensorflow BN implementation:
                node = tf.layers.batch_normalization(
                    node, training=is_training, name='BN', momentum=decay, trainable=trainable)

                # get variables:
                beta = tf.global_variables(tf.get_variable_scope().name + '/BN/beta')[0]
                gamma = tf.global_variables(tf.get_variable_scope().name + '/BN/gamma')[0]
                batch_mean = tf.global_variables(tf.get_variable_scope().name + '/BN/moving_mean')[0]
                batch_var = tf.global_variables(tf.get_variable_scope().name + '/BN/moving_variance')[0]

                # summary:
                if summary:
                    tf.summary.histogram('beta', beta)
                    tf.summary.histogram('gamma', gamma)
                    tf.summary.histogram('batch_mean', batch_mean)
                    tf.summary.histogram('batch_var', batch_var)

                # Record variables:
                if collections is not None:
                    tf.add_to_collections(collections, beta)
                    tf.add_to_collections(collections, gamma)
                    tf.add_to_collections(collections, batch_mean)
                    tf.add_to_collections(collections, batch_var)
            else:  # My own BN implementation:
                with tf.variable_scope(name_or_scope='BN'):
                    in_depth = node.shape.as_list()[-1]
                    axes = list(range(len(node.shape.as_list()) - 1))
                    if isinstance(is_training, bool):
                        is_training = tf.constant(is_training, tf.bool)

                    beta = tf.get_variable(name='beta', shape=[in_depth], dtype=tf.float32,
                                           initializer=tf.zeros_initializer(), trainable=trainable)
                    gamma = tf.get_variable(name='gamma', shape=[in_depth], dtype=tf.float32,
                                            initializer=tf.ones_initializer(), trainable=trainable)
                    batch_mean, batch_var = tf.nn.moments(node, axes=axes, name='moments')
                    ema = tf.train.ExponentialMovingAverage(decay=decay)

                    def train_fn():
                        update_ema = ema.apply([batch_mean, batch_var])
                        with tf.control_dependencies([update_ema]):
                            return tf.identity(batch_mean), tf.identity(batch_var)

                    def test_fn():
                        return ema.average(batch_mean), ema.average(batch_var)

                    train_flag = tf.logical_and(is_training, tf.constant(trainable, tf.bool))
                    mean, var = tf.cond(train_flag, train_fn, test_fn)
                    normalized = tf.nn.batch_normalization(node, mean, var, beta, gamma, 1e-3)
                    node = normalized

                    # summary:
                    if summary:
                        tf.summary.histogram('beta', beta)
                        tf.summary.histogram('gamma', gamma)
                        tf.summary.histogram('batch_mean', ema.average(batch_mean))
                        tf.summary.histogram('batch_var', ema.average(batch_var))

                    # Record variables:
                    if collections is not None:
                        tf.add_to_collections(collections, beta)
                        tf.add_to_collections(collections, gamma)
                        tf.add_to_collections(collections, ema.average(batch_mean))
                        tf.add_to_collections(collections, ema.average(batch_var))

        # ReLU:
        if use_ReLU:
            rl = tf.nn.relu(node, name='ReLU')
            node = rl

        # conv:
        with tf.variable_scope(name_or_scope='Conv'):
            conv = tf.nn.conv2d(node, w, strides=[1, stride, stride, 1], padding=pad_mode) + b

    return conv


def residual_unit(x,
                  w_a, w_b, b_a, b_b,
                  is_training,
                  name,
                  use_BN=True,
                  use_ReLU=True,
                  stride=1,
                  decay=0.99,
                  pad_mode='SAME',
                  summary=False,
                  trainable=True,
                  collections=None):
    """
    Define a residual unit in recursive block.

    :param x: Input tensor of this unit.
    :param w_a: Kernel of convolution A in this unit.
    :param w_b: Kernel of convolution B in this unit.
    :param b_a: Bias of convolution A in this unit.
    :param b_b: Bias of convolution B in this unit.
    :param is_training: Whether it is in training state. For BN layers.
    :param name: The name of this unit.
    :param use_BN: Whether to add BN layses in this residual unit.
    :param use_ReLU: Whether to add ReLU.
    :param stride: The stride in convolution layers.
    :param decay: The decay for ExponentialMovingAverage in BN layers.
    :param pad_mode: The padding mode in convolution layers.
    :param summary: Whether to generate a summary for the variables.
    :param trainable: Whether these variables are trainable.
    :param collections: The tags of collection lists of variables.
    :return: Output tensor of this unit.
    """
    with tf.variable_scope(name_or_scope=name):
        ru_a = BN_relu_conv(x=x, w=w_a, b=b_a, is_training=is_training, name='part_a',
                            use_BN=use_BN, use_ReLU=use_ReLU,
                            stride=stride, decay=decay, pad_mode=pad_mode, summary=summary,
                            trainable=trainable, collections=collections)
        ru_b = BN_relu_conv(x=ru_a, w=w_b, b=b_b, is_training=is_training, name='part_b',
                            use_BN=use_BN, use_ReLU=use_ReLU,
                            stride=stride, decay=decay, pad_mode=pad_mode, summary=summary,
                            trainable=trainable, collections=collections)
    return ru_b


def recursive_block(x,
                    count_U,
                    out_depth,
                    is_training,
                    name,
                    use_BN_at_begin=True,
                    use_BN_in_ru=True,
                    use_ReLU=True,
                    w_initializer=None,
                    b_initializer=None,
                    w_size=3,
                    stride=1,
                    decay=0.99,
                    pad_mode='SAME',
                    summary=False,
                    trainable=True,
                    collections=None):
    """
    Define a recursive block in DRRN.

    :param x: Input tensor of this block.
    :param count_U: The number of residual units in this block.
    :param out_depth: The number of channels.
    :param is_training: Whether it is in training state. For BN layers.
    :param name: The name of this block.
    :param use_BN_at_begin: Whether to add BN layses at the beginning of this recursive block.
    :param use_BN_in_ru: Whether to add BN layses in residual units.
    :param use_ReLU: Whether to add ReLU.
    :param w_initializer: The initializer for the variable w in convolution layers.
    :param b_initializer: The initializer for the variable b in convolution layers.
    :param w_size: The size of convolution kernels.
    :param stride: The stride in convolution layers.
    :param decay: The decay for ExponentialMovingAverage in BN layers.
    :param pad_mode: The padding mode in convolution layers.
    :param summary: Whether to generate a summary for the variables.
    :param trainable: Whether these variables are trainable.
    :param collections: The tags of collection lists of variables.
    :return: Output tensor of this block.
    """
    with tf.variable_scope(name_or_scope=name):
        # begin-conv:
        in_depth = x.shape.as_list()[-1]
        w_bgn = tf.get_variable(name='w_bgn', shape=[w_size, w_size, in_depth, out_depth], dtype=tf.float32,
                                initializer=w_initializer(), trainable=trainable)
        b_bgn = tf.get_variable(name='b_bgn', shape=[out_depth], dtype=tf.float32, initializer=b_initializer(),
                                trainable=trainable)

        # summary:
        if summary:
            tf.summary.histogram('w_bgn', w_bgn)
            tf.summary.histogram('b_bgn', b_bgn)

        # Record variables:
        if collections is not None:
            tf.add_to_collections(collections, w_bgn)
            tf.add_to_collections(collections, b_bgn)

        bgn = BN_relu_conv(x=x, w=w_bgn, b=b_bgn, is_training=is_training, name='bgn', use_BN=use_BN_at_begin,
                           use_ReLU=use_ReLU, stride=stride, decay=decay, pad_mode=pad_mode, summary=summary,
                           trainable=trainable, collections=collections)

        # residual units:
        node = bgn
        if count_U > 0:
            w_a = tf.get_variable(name='w_a', shape=[w_size, w_size, out_depth, out_depth], dtype=tf.float32,
                                  initializer=w_initializer(), trainable=trainable)
            b_a = tf.get_variable(name='b_a', shape=[out_depth], dtype=tf.float32, initializer=b_initializer(),
                                  trainable=trainable)
            w_b = tf.get_variable(name='w_b', shape=[w_size, w_size, out_depth, out_depth], dtype=tf.float32,
                                  initializer=w_initializer(), trainable=trainable)
            b_b = tf.get_variable(name='b_b', shape=[out_depth], dtype=tf.float32, initializer=b_initializer(),
                                  trainable=trainable)

            # summary:
            if summary:
                tf.summary.histogram('w_a', w_a)
                tf.summary.histogram('b_a', b_a)
                tf.summary.histogram('w_b', w_b)
                tf.summary.histogram('b_b', b_b)

            # Record variables:
            if collections is not None:
                tf.add_to_collections(collections, w_a)
                tf.add_to_collections(collections, b_a)
                tf.add_to_collections(collections, w_b)
                tf.add_to_collections(collections, b_b)

            for ru_num in range(count_U):
                node = bgn + residual_unit(node,
                                           w_a, w_b, b_a, b_b,
                                           is_training=is_training,
                                           name='RU_' + str(ru_num),
                                           use_BN=use_BN_in_ru,
                                           use_ReLU=use_ReLU,
                                           stride=stride,
                                           decay=decay,
                                           pad_mode=pad_mode,
                                           summary=summary,
                                           trainable=trainable,
                                           collections=collections)

    return node


def DRRN(x,
         count_B,
         count_U,
         out_depth,
         is_training,
         name=None,
         use_BN_at_begin=True,
         use_BN_in_ru=True,
         use_BN_at_end=True,
         use_ReLU=True,
         w_initializer=None,
         b_initializer=None,
         w_size=3,
         stride=1,
         decay=0.99,
         pad_mode='SAME',
         summary=False,
         trainable=True,
         collections=None):
    """
    DRRN inference define.

    :param x: Input tensor with size of [None, height, width, 1].
    :param count_B: The number of recursive blocks.
    :param count_U: The number of residual units.
    :param out_depth: The number of channels.
    :param is_training: Whether it is in training state. For BN layers.
    :param name: The net name. If is None, it will be: 'DRRN' + 'B' + str(count_B) + 'U' + str(count_U) + 'C' + str(out_depth)
    :param use_BN_at_begin: Whether to add BN layses at the beginning of each recursive block.
    :param use_BN_in_ru: Whether to add BN layses in residual unit.
    :param use_BN_at_end: Whether to add BN layses at the end of the DRRN net.
    :param use_ReLU: Whether to add ReLU.
    :param w_initializer: The initializer for the variable w in convolution layers. Default is tf.glorot_uniform_initializer.
    :param b_initializer: The initializer for the variable b in convolution layers. Default is tf.zeros_initializer.
    :param w_size: The size of convolution kernels.
    :param stride: The stride in convolution layers.
    :param decay: The decay for ExponentialMovingAverage in BN layers.
    :param pad_mode: The padding mode in convolution layers.
    :param summary: Whether to generate a summary for the variables.
    :param trainable: Whether these variables are trainable.
    :param collections: The tags of collection lists of variables.
    :return: Output tensor with size of [None, height, width, 1].
    """
    name = name or 'DRRN' + 'B' + str(count_B) + 'U' + str(count_U) + 'C' + str(out_depth)
    w_initializer = w_initializer or tf.glorot_uniform_initializer
    b_initializer = b_initializer or tf.zeros_initializer
    with tf.variable_scope(name_or_scope=name):
        # recursive blocks:
        node = x
        for rb_num in range(count_B):
            node = recursive_block(x=node,
                                   count_U=count_U,
                                   out_depth=out_depth,
                                   is_training=is_training,
                                   name='RB_' + str(rb_num),
                                   use_BN_at_begin=use_BN_at_begin,
                                   use_BN_in_ru=use_BN_in_ru,
                                   use_ReLU=use_ReLU,
                                   w_initializer=w_initializer,
                                   b_initializer=b_initializer,
                                   w_size=w_size,
                                   stride=stride,
                                   decay=decay,
                                   pad_mode=pad_mode,
                                   summary=summary,
                                   trainable=trainable,
                                   collections=collections)

        # end-conv:
        in_depth = node.shape.as_list()[-1]
        w_end = tf.get_variable(name='w_end', shape=[w_size, w_size, in_depth, 1], dtype=tf.float32,
                                initializer=w_initializer(), trainable=trainable)
        b_end = tf.get_variable(name='b_end', shape=[1], dtype=tf.float32, initializer=b_initializer(),
                                trainable=trainable)

        # summary:
        if summary:
            tf.summary.histogram('w_end', w_end)
            tf.summary.histogram('b_end', b_end)

        # Record variables:
        if collections is not None:
            tf.add_to_collections(collections, w_end)
            tf.add_to_collections(collections, b_end)

        end = BN_relu_conv(x=node, w=w_end, b=b_end, is_training=is_training, name='end', use_BN=use_BN_at_end,
                           use_ReLU=use_ReLU, stride=stride, decay=decay, pad_mode=pad_mode, summary=summary,
                           trainable=trainable, collections=collections)
        end = end + x

    return end


def MSE_float(predictions, labels):
    return tf.losses.mean_squared_error(labels=labels, predictions=predictions, scope='MSE_float')


def loss_L1(predictions, labels):
    return tf.losses.absolute_difference(labels=labels, predictions=predictions, scope='L1_loss')


def PSNR_float(mse):
    with tf.name_scope(name='PSNR_float'):
        psnr = tf.multiply(10.0, tf.log(1.0 * 1.0 / mse) / tf.log(10.0))
    return psnr


def Outputs(predictions):
    with tf.name_scope(name='Outputs'):
        results = tf.clip_by_value(tf.round(255.0 * predictions), 0, 255)
        results = tf.cast(results, tf.uint8)
    return results


def PSNR_uint8(outputs, labels):
    with tf.name_scope(name='PSNR_uint8'):
        with tf.name_scope(name='MSE'):
            mse = tf.reduce_mean(
                tf.square(tf.clip_by_value(tf.round(255.0 * labels), 0, 255) - tf.cast(outputs, tf.float32)))
        psnr = tf.multiply(10.0, tf.log(255.0 * 255.0 / mse) / tf.log(10.0))
    return psnr


# dataset ##################################################
def dataset_TFR(TFR_path, batch_size, patch_height, patch_width, shuffle=None):
    """
    Dataset from TFR files.
    """

    def example_process(exa):
        ims = tf.parse_single_example(exa, features={
            'input': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        })
        im_input = ims['input']
        im_label = ims['label']
        im_input = tf.decode_raw(im_input, tf.uint8)
        im_label = tf.decode_raw(im_label, tf.uint8)
        im_input = tf.reshape(im_input, [patch_height, patch_width, 1]) / 255
        im_label = tf.reshape(im_label, [patch_height, patch_width, 1]) / 255
        return {'input': im_input, 'label': im_label}

    dataset = tf.data.TFRecordDataset(TFR_path)
    if shuffle is not None:
        dataset = dataset.shuffle(buffer_size=shuffle)
    dataset = dataset.repeat()
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(map_func=example_process, batch_size=batch_size, num_parallel_batches=4))
    # dataset = dataset.map(example_process)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=16)
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    return batch


def dataset_YUV(input_path, label_path, batch_size, patch_height, patch_width, shuffle=True, repeat=None):
    """
    Dataset from YUV files.
    """
    input, _, _ = YUVread(input_path, [patch_height, patch_width])
    label, _, _ = YUVread(label_path, [patch_height, patch_width])

    input = input[:, :, :, np.newaxis, np.newaxis] / 255
    label = label[:, :, :, np.newaxis, np.newaxis] / 255

    data = np.concatenate((input, label), axis=4)

    if shuffle:
        np.random.shuffle(data)

    bgn = 0
    data_len = data.shape[0]

    while (repeat is None) or (repeat > 0):
        batch = np.array([], dtype=np.float32).reshape([0, patch_height, patch_width, 1, 2])
        end = bgn + batch_size
        while end >= data_len:
            batch = np.concatenate((batch, data[bgn:, :, :, :, :]))
            if repeat is not None:
                repeat -= 1
                if repeat <= 0:
                    yield {'input': batch[:, :, :, :, 0], 'label': batch[:, :, :, :, 1]}
                    return 'ALL DONE.'
            end = end - data_len
            bgn = 0
        batch = np.concatenate((batch, data[bgn:end, :, :, :, :]))
        bgn = end
        yield {'input': batch[:, :, :, :, 0], 'label': batch[:, :, :, :, 1]}

    return 'ALL DONE.'


if __name__ == '__main__':  # Run this file to get parameter-file size.
    print('logs: define grapth...\n')
    inputs = tf.placeholder(tf.float32, [None, 41, 41, 1])
    drrn = DRRN(inputs, 1, 9, 32, True, 'DRRNoverfit_B1U9C32', True, False, False, collections=['var_list'])
    # drrn = DRRN(inputs, 1, 9, 64, True, 'DRRNoverfit_B1U9C64', collections=['vars'])
    vars = tf.get_collection('var_list')
    calculate_variables(vars, True)
