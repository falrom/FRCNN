# 编码及数据处理

这个文件夹用于生产过拟合网络的训练数据，涉及到视频的编解码和相应的 YUV 文件操作，以及生成网络需要的 TFrecords 文件。

大体流程：从原始 YUV 测试码流中抽取 50 帧，使用 x265 工具进行 low_delay_P编码、解码，将原始 YUV 图像（训练标签）与解码重建的 YUV 图像（训练输入）切割、打包成 TFrecords 文件。

标准：

```
encoder version   : x265 2.6
encode parameters : --fps ${fps} --qp ${QP} --ipratio 1 --bframes 0
decoder version   : HM TAppDecoder 16.0
```

文件命名规范：

```
原始码流文件名 : <OrigName>.yuv
截取帧后文件名 : <OrigName>_<StartNum>to<EndNum>.yuv
编码码流文件名 : <OrigName>_<StartNum>to<EndNum>_QP<QP>_<mode>.bin
解码码流文件名 : <OrigName>_<StartNum>to<EndNum>_QP<QP>_<mode>_rec.yuv
TFrecords文件名 : <OrigName>_<StartNum>to<EndNum>_QP<QP>_<mode>_<PatchSize>x<PatchSize>.tfrecords
```

以`BasketballDrive_1920x1080_50.yuv`码流为例：
```
原始码流文件名 : BasketballDrive_1920x1080_50.yuv
截取帧后文件名 : BasketballDrive_1920x1080_50_000to049.yuv  # 即截取了前50帧
编码码流文件名 : BasketballDrive_1920x1080_50_000to049_QP25_IP.bin  # IP: low_delay_P
解码码流文件名 : BasketballDrive_1920x1080_50_000to049_QP25_IP_rec.yuv
TFrecords文件名 : BasketballDrive_1920x1080_50_000to049_QP25_IP_41x41.tfrecords
```

## 生产数据步骤：

两种方法。

### 一步到位

在`generate_dataset.py`文件中修改填写信息，运行即可。

### 分三步走

#### 截取帧

在`yuv_io.py`文件中设置**码流文件名**、**分辨率**、**帧截取位置**、**截取帧数**，运行该文件即可：

```bash
python yuvio.py
```

#### 编解码

在`codex.bat`文件中编辑**码流文件名**、**码流FPS**、**编码QP**，运行该文件即可。

编码器使用的是编译好的 x265 工具，解码使用的是 HM 编译出来的解码器。编码、解码的 log 分别存储在了`decode.log`和`encode.log`文件中。

#### 切片并生成 TFrecords 文件

在`cut_TFR.py`文件中设置**输入文件名**、**编码QP**、**切片尺寸信息**，运行该文件即可。生成的 TFrecords 文件在`./TFRdata`目录下。

*\* Number of patches: 76850 (For BasketballDrive_1920x1080_50_000to049 patch_size=41 step=36
height=1080 width=1920)*


