import os
from tqdm import tqdm
import json

import numpy as np

import onnx
from onnx_tf.backend import prepare
from onnx2keras import onnx_to_keras
from torch.autograd import Variable
from pytorch2keras import pytorch_to_keras

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data.distributed
from torchvision import datasets, transforms

from mcunet.model_zoo import build_model
from mcunet.utils import AverageMeter, accuracy, count_net_flops, count_parameters

from tensorflow import keras
import tensorflow as tf

from keras.datasets import cifar10
from keras.utils import to_categorical

# Training settings
parser = argparse.ArgumentParser()
# net setting
parser.add_argument('--net_id', type=str, help='net id of the model')
# data loader setting
parser.add_argument('--dataset', default='imagenet', type=str, choices=['imagenet', 'vww'])
parser.add_argument('--data-dir', default=os.path.expanduser('/dataset/imagenet/val'),
                    help='path to ImageNet validation data')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = 'cuda'


def build_val_data_loader(resolution):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    kwargs = {'num_workers': args.workers, 'pin_memory': True}

    if args.dataset == 'imagenet':
        val_transform = transforms.Compose([
            transforms.Resize(int(resolution * 256 / 224)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize
        ])
    elif args.dataset == 'vww':
        val_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),  # if center crop, the person might be excluded
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise NotImplementedError
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    return val_loader


def validate(model, val_loader):
    model.eval()
    val_loss = AverageMeter()
    val_top1 = AverageMeter()

    with tqdm(total=len(val_loader), desc='Validate') as t:
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                val_loss.update(F.cross_entropy(output, target).item())
                top1 = accuracy(output, target, topk=(1,))[0]
                val_top1.update(top1.item(), n=data.shape[0])
                t.set_postfix({'loss': val_loss.avg,
                               'top1': val_top1.avg})
                t.update(1)

    return val_top1.avg

def use_onnx_to_keras(model, rand_input):
    model(rand_input)
    print("success with rand_input")

    # Set input and output names, include more names in the list if your model has more than 1 input/output
    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}

    # Export model with the above parameters
    torch_out = torch.onnx.export(
    model, rand_input, 'model.onnx', export_params=True, input_names=input_names, output_names=output_names, 
    dynamic_axes=dynamic_axes, operator_export_type=torch.onnx.OperatorExportTypes.ONNX
    )

    # Use ONNX checker to verify integrity of model
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
    print("success with onnx_model")

    onnx_model = onnx.load('model.onnx')
    tf_model = prepare(onnx_model)
    tf_model.export_graph('tf_model') # saved in the tf_model folder
    print("success with export")

    # tf.keras.models.save_model(tf_model.tf_module, 'mcunetv1.h5')
    # pb_model = tf.keras.models.load_model("./tf_model")

    # import pdb
    # pdb.set_trace()
    # tf.keras.models.save_model(pb_model, './mcunetv1.h5')

def onnx_to_h5():
    onnx_model = onnx.load('model.onnx')

    import pdb
    pdb.set_trace()

    k_model = onnx_to_keras(onnx_model, ['input'])


def main():
    model, resolution, description = build_model(args.net_id, pretrained=True)
    # model = model.to(device)
    model.eval()
    val_loader = build_val_data_loader(resolution)

    # profile model
    total_macs = count_net_flops(model, [1, 3, resolution, resolution])
    total_params = count_parameters(model)
    print(' * FLOPs: {:.4}M, param: {:.4}M'.format(total_macs / 1e6, total_params / 1e6))

    # acc = validate(model, val_loader)
    # print(' * Accuracy: {:.2f}%'.format(acc))

    # #onnx prepare
    # rand_input = torch.randn((1, 3, resolution, resolution), requires_grad=True)
    # use_onnx_to_keras(model, rand_input)

    # onnx_to_h5()

    input_np = np.random.uniform(0, 1, (1, 3, resolution, resolution))
    input_var = Variable(torch.FloatTensor(input_np))
    k_model = pytorch_to_keras(model, input_var, [(3, resolution, resolution,)], verbose=True, change_ordering=True)

    print("finish here")

    import pdb
    pdb.set_trace()




if __name__ == '__main__':
    main()
