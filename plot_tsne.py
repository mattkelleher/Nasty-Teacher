# plot tSNE for a model
# May 02, 2021, Sameer Bibikar, modified from train_scratch.py

import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from tqdm import tqdm
import argparse
import os
import logging
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from utils.utils import RunningAverage, set_logger, Params
from model import *
from data_loader import fetch_dataloader


# ************************** random seed **************************
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='experiments/CIFAR10/baseline/resnet18', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--no-cuda', dest='cuda', action='store_false', default=True)
parser.add_argument('--layer-size', default=None, type=int)
args = parser.parse_args()

device_ids = args.gpu_id
torch.cuda.set_device(device_ids[0])


def plot_tsne(model, loss_fn, data_loader, params):
    model.eval()
    # summary for current eval loop
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        # compute metrics over the dataset
        for i, (data_batch, labels_batch) in enumerate(tqdm(data_loader)):
            if params.cuda:
                data_batch = data_batch.cuda()          # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output
            output_batch = model(data_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            all_outputs.append(output_batch)
            labels_batch = labels_batch.cpu().numpy()
            all_labels.append(labels_batch)

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    embedded = TSNE(learning_rate=1000, n_iter=1000, perplexity=20, n_jobs=-1).fit_transform(all_outputs)

    fig, ax = plt.subplots()
    ax.scatter(embedded[:, 0], embedded[:, 1], s=1, c=all_labels, cmap='tab10', alpha=0.5)
    return fig, ax



if __name__ == "__main__":
    # ************************** set log **************************
    set_logger(os.path.join(args.save_path, 'plot_tsne.log'))

    # #################### Load the parameters from json file #####################################
    json_path = os.path.join(args.save_path, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    params.cuda = torch.cuda.is_available() and args.cuda # use GPU if available

    for k, v in params.__dict__.items():
        logging.info('{}:{}'.format(k, v))

    # ########################################## Dataset ##########################################
    trainloader = fetch_dataloader('train', params)
    devloader = fetch_dataloader('dev', params)

    # ############################################ Model ############################################
    if params.dataset == 'cifar10':
        num_class = 10
    elif params.dataset == 'cifar100':
        num_class = 100
    elif params.dataset == 'tiny_imagenet':
        num_class = 200
    else:
        num_class = 10

    logging.info('Number of class: ' + str(num_class))
    logging.info('Create Model --- ' + params.model_name)

    # ResNet 18 / 34 / 50 ****************************************
    if params.model_name == 'resnet18':
        model = ResNet18(num_class=num_class)
    elif params.model_name == 'resnet34':
        model = ResNet34(num_class=num_class)
    elif params.model_name == 'resnet50':
        model = ResNet50(num_class=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif params.model_name.startswith('preresnet20'):
        model = PreResNet(depth=20, num_classes=num_class)
    elif params.model_name.startswith('preresnet32'):
        model = PreResNet(depth=32, num_classes=num_class)
    elif params.model_name.startswith('preresnet44'):
        model = PreResNet(depth=44, num_classes=num_class)
    elif params.model_name.startswith('preresnet56'):
        model = PreResNet(depth=56, num_classes=num_class)
    elif params.model_name.startswith('preresnet110'):
        model = PreResNet(depth=110, num_classes=num_class)

    # DenseNet *********************************************
    elif params.model_name == 'densenet121':
        model = densenet121(num_class=num_class)
    elif params.model_name == 'densenet161':
        model = densenet161(num_class=num_class)
    elif params.model_name == 'densenet169':
        model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif params.model_name == 'resnext29':
        model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif params.model_name == 'mobilenetv2':
        model = MobileNetV2(class_num=num_class)

    elif params.model_name == 'shufflenetv2':
        model = shufflenetv2(class_num=num_class)

    # Basic neural network ********************************
    elif params.model_name == 'net':
        model = Net(num_class, params)

    elif params.model_name == 'mlp':
        model = MLP(num_class=num_class)

    else:
        model = None
        print('Not support for model ' + str(params.model_name))
        exit()



    if 'teacher_deeper' in params.dict:
        if args.layer_size is None:
            args.layer_size = params.teacher_deeper
        if params.teacher_deeper:
            in_features = model.linear.in_features
            model.linear = nn.Linear(in_features, args.layer_size)
            model = nn.Sequential(model, nn.Linear(args.layer_size, num_class))

    if params.cuda:
        model = model.cuda()

    if len(args.gpu_id) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    # checkpoint ********************************
    if args.resume:
        logging.info('- Load checkpoint model from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info('- Plot tSNE ')

    # ************************** LOSS **************************
    criterion = nn.CrossEntropyLoss()

    # ################################# train and evaluate #################################
    fig, ax = plot_tsne(model, criterion, trainloader, params)
    fig.savefig(os.path.join(args.save_path, 'tsne.png'))


