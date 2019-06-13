import tensorflow as tf
import os
import sys
import numpy as np
from yuv_io import YUVread
from yuv_io import YUVwrite
from cut_TFR import progress_bar
from cut_TFR import generate


def generate_dataset(
        path_source,  # './videos/RaceHorses_832x480_30.yuv'
        cfg_name,
        QP,
        frame_bgn,
        frame_num,
        height=None,
        width=None,
        fps=None,
        patch_size=41,  # 31  # 41
        step=36  # 21  # 36
):
    OrigName = os.path.splitext(os.path.split(path_source)[-1])[0]
    _, size_str, fps_str = OrigName.split('_')[:3]
    fps = fps or int(fps_str)
    width_str, height_str = size_str.split('x')
    height = height or int(height_str)
    width = width or int(width_str)

    # Paths ##################################################################
    # Naming conventions:
    # 原始码流文件名   : <OrigName>.yuv
    # 截取帧后文件名   : <OrigName>_<StartNum>to<EndNum>.yuv
    # 编码码流文件名   : <OrigName>_<StartNum>to<EndNum>_QP<QP>_<mode>.bin
    # 解码码流文件名   : <OrigName>_<StartNum>to<EndNum>_QP<QP>_<mode>_rec.yuv
    # TFrecords文件名 : <OrigName>_<StartNum>to<EndNum>_QP<QP>_<mode>_<PatchSize>x<PatchSize>.tfrecords
    # Example:
    # 原始码流文件名   : BasketballDrive_1920x1080_50.yuv
    # 截取帧后文件名   : BasketballDrive_1920x1080_50_000to049.yuv  # 即截取了前50帧
    # 编码码流文件名   : BasketballDrive_1920x1080_50_000to049_QP25_IP.bin  # IP: low_delay_P
    # 解码码流文件名   : BasketballDrive_1920x1080_50_000to049_QP25_IP_rec.yuv
    # TFrecords文件名 : BasketballDrive_1920x1080_50_000to049_QP25_IP_41x41.tfrecords

    StartNum = str('%03d' % frame_bgn)
    EndNum = str('%03d' % (frame_bgn + frame_num - 1))
    mode = 'IP'

    path_cut = './videos/' + OrigName + '_' + StartNum + 'to' + EndNum + '.yuv'
    path_bin = './videos/' + OrigName + '_' + StartNum + 'to' + EndNum + '_QP' + str(QP) + '_' + mode + '_HM.bin'
    path_rec = './videos/' + OrigName + '_' + StartNum + 'to' + EndNum + '_QP' + str(QP) + '_' + mode + '_rec_HM.yuv'
    path_tfr = './TFRdata/' + OrigName + '_' + StartNum + 'to' + EndNum + '_QP' + str(QP) + '_' + mode + '_' + \
               str(patch_size) + 'x' + str(patch_size) + '_HM.tfrecords'

    # Cut yuv frames ##################################################################
    print('\n·········· Cut YUV frames ··········')
    print('Input :', path_source)
    print('Output:', path_cut)
    if not os.path.exists(path_cut):
        print('Processing...')
        y, u, v = YUVread(path=path_source, size=[height, width], frame_num=frame_num, start_frame=frame_bgn)
        YUVwrite(y, u, v, path=path_cut)
        print('Done.')
    else:
        print('Already exists.')

    # Codex ##################################################################
    print('\n·········· Codec ··········')
    print('Source YUV:', path_cut)
    print('Target bin:', path_bin)
    print('Decode YUV:', path_rec)
    print('Encoding...')
    # os.system(
    #     'x265.exe --input-res ' + str(width) + 'x' + str(height) + ' --fps ' + str(fps) + ' ' + path_cut + ' -o ' +
    #     path_bin + ' --qp ' + str(QP) + ' --ipratio 1 --bframes 0 --psnr --ssim 1>>encode.log 2>&1')
    # print('Decoding...')
    # os.system('TAppDecoder.exe -b ' + path_bin + ' -o ' + path_rec + ' 1>>decode.log 2>&1')
    os.system('TAppEncoder.exe -c encoder_lowdelay_P_main.cfg -c ' + cfg_name + ' -q ' + str(QP) + ' -b ' + path_bin + ' -o ' + path_rec + ' -f ' + str(frame_num) + ' > ' + OrigName + '_' + 'Encode_log_QP' + str(QP) +'.txt')
    print('Done.')

    # Generate TFrecords files ##################################################################
    target_dir = 'TFRdata/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print('\n·········· Cut patches ··········')
    print('Input  path:', path_rec)
    print('Label  path:', path_cut)
    print('Output path:', path_tfr)
    writer = tf.python_io.TFRecordWriter(path_tfr)

    print('Reading files...', end=' ')
    sys.stdout.flush()
    im_input, _, _ = YUVread(path_rec, [height, width])
    im_label, _, _ = YUVread(path_cut, [height, width])
    print('Done.', flush=True)

    number = im_input.shape[0]
    count_patch = 0
    for count in range(number):
        count_patch += generate(im_input[count], im_label[count], [height, width], patch_size, step, writer)
        progress_bar(count + 1, number)

    print('\nNumber of patches:', count_patch)
    writer.close()


if __name__ == '__main__':
    videos = [
        './videos/BasketballDrive_1920x1080_50.yuv',
        './videos/BQTerrace_1920x1080_60.yuv',
        './videos/Cactus_1920x1080_50.yuv',
        './videos/Kimono1_1920x1080_24.yuv',
        './videos/ParkScene_1920x1080_24.yuv'
    ]

    video_config = [
        './video_config/BasketballDrive.cfg',
        './video_config/BQTerrace.cfg',
        './video_config/Cactus.cfg',
        './video_config/Kimono.cfg',
        './video_config/ParkScene.cfg',
    ]



    QPs = [22, 27 ,32, 37]
    frame_bgn = 0
    frame_num = 50

    for i in range(5):
        for QP in QPs:
            print('\n=========================================================================')
            generate_dataset(path_source=videos[i], cfg_name=video_config[i], QP=QP, frame_bgn=frame_bgn, frame_num=frame_num)
