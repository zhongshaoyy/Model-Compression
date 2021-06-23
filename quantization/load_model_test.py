import torch
from models.model_cifar.resnet_small import *
# from models.model_cifar.resnet import *
import time
import torch.nn as nn
import models
import os, argparse
from utils import *

default_cfg_cifar10 = {
    '18': [16, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64],
    '20': [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64],
    '32': [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
           32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
           64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    '56': [16,
           16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
           32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
           64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    '110': [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
}

# Set parameters
parser = argparse.ArgumentParser()

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser.add_argument('--model', metavar='ARCH', default='resnet56', type=str,
                    help='Student model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Input the dataset name: default(CIFAR10)')
parser.add_argument('--data_path', default='/home/ran/mnt1/Dataset/data.cifar10',
                    type=str, help='Input the dataset name: default(CIFAR10)')
parser.add_argument('--num_workers', default=4, type=int, help='Input the number of works: default(0)')
parser.add_argument('--num_calibration_batches', default=10,
                    type=int, help='Input the number of epoches: default(300)')
parser.add_argument('--train_batch_size', default=128, type=int, help='Input the batch size: default(128)')
parser.add_argument('--eval_batch_size', default=128, type=int, help='Input the batch size: default(128)')
parser.add_argument('--type', default='Post', choices=['Post', 'PerChannel'],
                    type=str, help='Input the type of quantization: default(V0)')
parser.add_argument('--checkpoint_path', default='checkpoints/CIFAR10/resnet20/resnet_original.pth.tar',
                    type=str, help='Input the path of pretrained model: default('')')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(args)

# checkpoint = torch.load('/home/ran/mnt1/2020bnan/resnet-cifar10-best/pruned/resnet_bnat20_cifar10_pruned_best_2020-04-15_acc91.03/resnet_small20_cifar10_acc90.97.pth.tar', map_location='cpu')
# cfg = checkpoint['cfg']
# model = resnet20_small(cfg=cfg, KD=True, inter_layer=True)
# model.load_state_dict(checkpoint['state_dict'])
# # torch.autograd.set_detect_anomaly(True)
# # with torch.autograd.set_detect_anomaly(True):
# x = torch.rand(1,3,32,32)
# ff,f,logit = model(x)
# # logit.backward(torch.ones(1, 10).to('cpu'))
# # print(model)

checkpoint = torch.load('/home/ran/mnt1/ZJU-thesis/resnet-cifar10/56/teacher.pth.tar', map_location='cpu')


def load_resnet_state(model, checkpoint):
    mystate = model.state_dict()
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    for k1, k2 in zip(mystate.keys(), checkpoint.keys()):
        mystate[k1] = checkpoint[k2]
    model.load_state_dict(mystate)

model_folder = "model_cifar"
model_fd = getattr(models, model_folder)
model_cfg = getattr(model_fd, 'resnet')
model = getattr(model_cfg, 'resnet56' + '_small')(cfg=default_cfg_cifar10['56'], num_classes=10)
# model = resnet20_small(cfg=checkpoint['cfg'])
load_resnet_state(model, checkpoint)

model.to('cpu')
model.eval()
model.fuse_model()
print('\n After fusion and quantization, note fused modules: \n\n {}'.format(model.features[0]))

print('Size of baseline model (MB): {}'.format(os.path.getsize('/home/ran/mnt1/ZJU-thesis/resnet-cifar10/56/teacher.pth.tar')/1e6))

# num_eval_batches = 10
data_loader, data_loader_test = cifar_data_loaders(args.data_path, args.train_batch_size,
                                                   args.eval_batch_size, device='cpu')
criterion = nn.CrossEntropyLoss()

# top1, top5 = evaluate(model, criterion, data_loader_test)
# print('Evaluation Original accuracy, Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} '.format(top1=top1, top5=top5))
torch.jit.save(torch.jit.script(model),
               os.path.join('/home/ran/mnt1/ZJU-thesis/resnet-cifar10/56/', args.model + '_float_scripted.pth.tar'))



def run_benchmark(model_file, img_loader, type='script', num_batches=50):
    '''     测试量化模型的CPU加速效果    '''
    elapsed = 0
    if type == 'script':
        model = torch.jit.load(model_file)       # 对比速度
    else:
        model = torch.load(model_file)

    print('Loaded Model: {}'.format(model_file))
    print('Start to test run-time on CPU ... ')
    model.eval()
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            _ = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.5f ms' % (elapsed/num_images*1000))
    return elapsed

run_benchmark(os.path.join('/home/ran/mnt1/ZJU-thesis/resnet-cifar10/56/', args.model + '_float_scripted.pth.tar'),
              data_loader_test)
run_benchmark(os.path.join('/home/ran/mnt1/ZJU-thesis/Quantized_torch/results/CIFAR10/200QAT/resnet56_2020-12-07_22-12-50',
                           'resnet56_qat_scripted.pth.tar'),
              data_loader_test)