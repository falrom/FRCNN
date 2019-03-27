# FRCNN

Enhanced video coding system with online training neural network.

## Abstract

We herein propose an efficient video coding system with online training neural network to improve coding efficiency. A frame restoration convolutional neural network (FRCNN) is trained for each group of pictures of each sequence to repair the quality of each reconstructed frame. Using only the current encoding video stream as a training set, the FRCNN can restore the reconstructed frames very meticulously. Even at low bit rates, the final output of the FRCNN can improve the video quality effectively. Moreover, an efficient parameter coding scheme is applied to compress the parameters of the online training FRCNN. Subsequently, the compressed bits are transmitted to the decoder as part of the encoded bitstream. Compared with the latest High Efficiency Video Coding standard video coding, the proposed system can achieve 3.8–14.0% Bjøntegaard-Delta rate reduction, which is much higher than most of the existing neural-network-based video coding systems. The restoration network will be an additional part of the traditional standard codec without any structure change, thereby rendering it compatible with the existing coding systems.

## How to run the project

platform: Tensorflow 1.10 @ Python 3.6

### Prepare dataset

More information in `./data/README.md`.

### Train & test FRCNN

Run file `run.py` to train or evaluate the FRCNN after dataset is ready.

To train the FRCNN:

```bash
python run.py -g 0 train -b 1 -u 9 -c 32 -v BasketballDrive_1920x1080_50_000to049 -q 37 --max_steps 300000 --no_BN_ru --no_BN_end --L1
```

To evaluate the FRCNN:

```bash
???
```

Or type these lines in shell for more help:

```bash
python run.py -h
python run.py train -h
python run.py evaluate -h
```

### Compress

Compressing network weights involves two steps: **weights precision reduction** and **Huffman coding**.

#### weights precision reduction

#### Huffman coding











