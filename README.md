# model compress
Implementation with PyTorch. 

复现人：刘宇昂

## Data
### ImageNet
Download the ImageNet dataset from [here](http://image-net.org/download-images).
No need to split the val set into corresponding folders because we provide api codes to load data:
```
from mydataset.imagenet_dataset import ImageNetDataset
```
### CIFAR10
Use ```torchvision.datasets.CIFAR10()```

## Contents
model-compress


|-- pruning     剪枝

|-- distillation    蒸馏

|-- quantization      量化

