import tensorflow as tf
from yuv_io import *
import os
import sys


def progress_bar(num, total, width=40):
    rate = num / total
    rate_num = int(rate * width)
    r = '\r[%s%s] %d%%%s%d' % ("=" * rate_num, " " * (width - rate_num), int(rate * 100), ' done of ', total)
    sys.stdout.write(r)
    sys.stdout.flush()


def generate(im_input, im_label, im_size, patch_size, step, writer):
    [hgt, wdt] = im_size
    count_h = int((hgt - patch_size) / step + 1)
    count_w = int((wdt - patch_size) / step + 1)

    start_h = 0
    for h in range(count_h):
        start_w = 0
        for w in range(count_w):
            patch_input = im_input[start_h:start_h + patch_size, start_w:start_w + patch_size]
            patch_label = im_label[start_h:start_h + patch_size, start_w:start_w + patch_size]
            example = tf.train.Example(features=tf.train.Features(feature={
                'input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_input.tostring()])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_label.tostring()]))
            }))
            writer.write(example.SerializeToString())
            start_w += step
        start_h += step
    return count_h * count_w


if __name__ == '__main__':
    file_name = 'BasketballDrive_1920x1080_50_000to049'
    QP = 22
    height = 1080
    width = 1920
    mode = 'IP'

    patch_size = 41  # 31  # 41
    step = 36  # 21  # 36

    source_dir = 'videos/'
    target_dir = 'TFRdata/'
    input_file = source_dir + file_name + '_QP' + str(QP) + '_' + str(mode) + '_rec.yuv'
    label_file = source_dir + file_name + '.yuv'
    target_path = os.path.join(target_dir,
                               file_name + '_QP' + str(QP) + '_' + str(mode) + '_' + str(patch_size) + 'x' + str(
                                   patch_size) + '.tfrecords')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print('\n·········· Cut patches ··········')
    print('input  path:', input_file)
    print('label  path:', label_file)
    print('output path:', target_path)
    writer = tf.python_io.TFRecordWriter(target_path)

    print('Reading files...', end=' ')
    sys.stdout.flush()
    im_input, u, v = YUVread(input_file, [height, width])
    im_label, u, v = YUVread(label_file, [height, width])
    print('Done.', flush=True)

    number = im_input.shape[0]
    count_patch = 0
    for count in range(number):
        count_patch += generate(im_input[count], im_label[count], [height, width], patch_size, step, writer)
        progress_bar(count + 1, number)

    print('\nNumber of patches:', count_patch)
    writer.close()
