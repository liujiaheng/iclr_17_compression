# Pytorch Reimplementation for [End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704)
## Running Script
```
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config.json -n baseline
```


## Data Preparation

We need to first prepare the training and validation data.
The trainging data is from flicker.com.
You can obtain the training data according to description of [CompressionData](https://github.com/liujiaheng/CompressionData)

The training details is similar to our another repo [compression](https://github.com/liujiaheng/compression)
