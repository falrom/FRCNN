# FRCNN

Enhanced video coding system with online training neural network.

## Abstract

We herein propose an efficient video coding system with online training neural network to improve coding efficiency. A frame restoration convolutional neural network (FRCNN) is trained for each group of pictures of each sequence to repair the quality of each reconstructed frame. Using only the current encoding video stream as a training set, the FRCNN can restore the reconstructed frames very meticulously. Even at low bit rates, the final output of the FRCNN can improve the video quality effectively. Moreover, an efficient parameter coding scheme is applied to compress the parameters of the online training FRCNN. Subsequently, the compressed bits are transmitted to the decoder as part of the encoded bitstream. Compared with the latest High Efficiency Video Coding standard video coding, the proposed system can achieve 3.8–14.0% Bjøntegaard-Delta rate reduction, which is much higher than most of the existing neural-network-based video coding systems. The restoration network will be an additional part of the traditional standard codec without any structure change, thereby rendering it compatible with the existing coding systems.

## How to run the project

Platform: Tensorflow 1.10 @ Python 3.6

### Prepare dataset

More information in `./data/README.md`.

### Train & test FRCNN

Run file `run.py` to train or evaluate the FRCNN after dataset is ready.

To train the FRCNN:

```bash
python run.py -g 0 train -b 1 -u 9 -c 32 -v BasketballDrive_1920x1080_50_000to049 -q 37 --max_steps 500000 --no_BN_ru --no_BN_end --L1 --lr 0.002 --decay 0.999995
```

#### arguments

> -g 0: GPU device selection  
> train: run mode  
> -b 1: use 1 recursive block in DRRN  
> -u 9: use 9 residual units in every recursive block  
> -c 32: channel number is 32 in CNN  
> -v Basket...000to049: video name  
> -q 37: QP number in codex  
> --max_steps 300000: train steps  
> --no_BN_ru: disable the BN layers in residual units  
> --no_BN_end: disable the BN layers in the end convolution layer  
> --L1: use L1 loss, not MSE  
> --decay: learning rate decay: lr = lr * (decay^step)

To evaluate the FRCNN:

```bash
python run.py -g 0 evaluate -b 1 -u 9 -c 32 -v BasketballDrive_1920x1080_50_000to049 -q 37 --height 1080 --width 1920 --no_BN_ru --no_BN_end --ckpt checkpoints/DRRNoverfit_B1U9C32/BasketballDrive_1920x1080_50_000to049_QP37/20190330123736/20190331024130-400000
```

Or type these lines in shell for more help:

```bash
python run.py -h
python run.py train -h
python run.py evaluate -h
```

### Compress

Compressing network weights involves two steps: **weights precision reduction** and **Huffman coding**, which are written in file `utils.py` and file `huffman.py`, respectively.

Here is a one-click script `compress.py`:

```bash
python compress.py -i ./checkpoints/DRRNoverfit_B1U9C32/BasketballDrive_1920x1080_50_000to049_QP32/20190407223926/20190408171134-500000
```

Several files will be generated:

> \# Original checkpoint:  
> 20190408171134-500000.data-00000-of-00001  
> 20190408171134-500000.index
>
> \# 16bit checkpoint:  
> 20190408171134-500000-16bit.data-00000-of-00001  
> 20190408171134-500000-16bit.index
>
> \# 16big-Huffman checkpoint:  
> 20190412075210-500000-16bit-hfm.data-00000-of-00001
>
> \# 16big-deHuffman checkpoint:  
> 20190412075210-500000-16bit-hfm-dehfm.data-00000-of-00001  
> 20190412075210-500000-16bit-hfm-dehfm.index
>
> \# Final 32bit checkpoint:  
> 20190412075210-500000-16bit-hfm-dehfm-32bit.data-00000-of-00001  
> 20190412075210-500000-16bit-hfm-dehfm-32bit.index

The *16big-Huffman checkpoint* (`20190412075210-500000-16bit-hfm.data-00000-of-00001`) is the file that needs to be transferred eventually.

The *Final 32bit checkpoint* (`20190412075210-500000-16bit-hfm-dehfm-32bit.data-00000-of-00001` & `20190412075210-500000-16bit-hfm-dehfm-32bit.index`) is the one you should evaluate with.









