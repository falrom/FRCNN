from tensorflow.python import pywrap_tensorflow
from utils import draw_weight
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    ckpt_path = r'checkpoints\DRRNoverfit_B1U9C32\BasketballDrive_1920x1080_50_000to049_QP22\20190411171903\20190412075210-500000'

    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    for key in var_to_shape_map:
        if 'Adam' in key: continue
        if 'w_' in key:
            print('=' * 40)
            print('Tensor  name:', key)
            ckpt_data = np.float64(np.array(reader.get_tensor(key)))  # cast list to np arrary
            print('Tensor shape:', ckpt_data.shape)
            draw_weight(ckpt_data, title=key)
    plt.show()
    os.system('PAUSE')
