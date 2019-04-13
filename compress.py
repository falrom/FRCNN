from utils import transform_ckpt_bit_width
from huffman import HuffmanCodec
import argparse
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Checkpoint path.')
    args = parser.parse_args()

    name_orig = args.input
    name_16bit = name_orig + '-16bit'
    name_huffman = name_16bit + '-hfm'
    name_dehuffman = name_huffman + '-dehfm'
    name_32bit = name_dehuffman + '-32bit'

    ext = '.data-00000-of-00001'
    path_16bit = name_16bit + ext
    path_huffman = name_huffman + ext
    path_dehuffman = name_dehuffman + ext
    path_dehuffman_index = name_dehuffman + '.index'

    transform_ckpt_bit_width(name_orig, name_16bit, '16')
    hc = HuffmanCodec()
    hc.compress(path_16bit, path_huffman)
    hc.uncompress(path_huffman, path_dehuffman)
    shutil.copyfile(name_16bit + '.index', path_dehuffman_index)
    transform_ckpt_bit_width(name_dehuffman, name_32bit, '32')
