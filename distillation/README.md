
## Structure of Repository
```
├── cifar_config.py  # Hyperparameters
├── cifar_train.py
├── data
│   └── directory_of_data.md
├── imagenet_config.py  # Hyperparameters
├── imagenet_train.py
├── losses
│   ├── cd_loss.py  # CD Loss
│   ├── ce_loss.py
│   ├── __init__.py
│   └── kd_loss.py  # GKD Loss
├── models
│   ├── channel_distillation.py  # Distillation Network
│   ├── __init__.py
│   └── resnet.py
├── pretrain
│   └── path_of_teacher_checkpoint.md
├── README.md
└── utils
    ├── average_meter.py
    ├── data_prefetcher.py
    ├── __init__.py
    ├── logutil.py
    ├── metric.py
    └── util.py  # Early Decay Teacher
```

## Requirements

> python >= 3.7  
> torch >= 1.4.0  
> torchvision >= 0.5.0

## Experiments

### ImageNet

#### Prepare Dataset

+ Download the ImageNet dataset from http://www.image-net.org/
+ Then, move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

```bash
images should be arranged in this way

./data/ILSVRC2012/train/dog/xxx.png
./data/ILSVRC2012/train/cat/xxy.png
./data/ILSVRC2012/val/dog/xxx.png
./data/ILSVRC2012/val/cat/xxy.png
```

#### Training

Note  
> Teacher checkpoint will be downloaded automatically.  

Running the following command and experiment will be launched.

```bash
CUDA_VISIBLE_DEVICES=0 python3 ./imagenet_train.py
```

If you want to run other experiments, you just need modify following losses in `imagenet_config.py`

+ s_resnet18.t_resnet34.cd.ce
```python
loss_list = [
    {"loss_name": "CELoss", "loss_rate": 1, "factor": 1, "loss_type": "ce_family", "loss_rate_decay": "lrdv1"},
    {"loss_name": "CDLoss", "loss_rate": 6, "factor": 1, "loss_type": "fd_family", "loss_rate_decay": "lrdv1"},
]
```

+ s_resnet18.t_resnet34.cd.ce.kdv2
```python
loss_list = [
    {"loss_name": "CELoss", "loss_rate": 1, "factor": 1, "loss_type": "ce_family", "loss_rate_decay": "lrdv1"},
    {"loss_name": "KDLossv2", "T": 1, "loss_rate": 1, "factor": 1, "loss_type": "kdv2_family", "loss_rate_decay": "lrdv1"},
    {"loss_name": "CDLoss", "loss_rate": 6, "factor": 0.9, "loss_type": "fd_family", "loss_rate_decay": "lrdv1"},
]
```

+ s_resnet18.t_resnet34.cd.kdv2.lrdv2
```python
loss_list = [
    {"loss_name": "CELoss", "loss_rate": 1, "factor": 1, "loss_type": "ce_family", "loss_rate_decay": "lrdv2"},
    {"loss_name": "KDLossv2", "T": 1, "loss_rate": 1, "factor": 1, "loss_type": "kdv2_family", "loss_rate_decay": "lrdv2"},
    {"loss_name": "CDLoss", "loss_rate": 6, "factor": 0.9, "loss_type": "fd_family", "loss_rate_decay": "lrdv2"},
]
```



