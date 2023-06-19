# Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import argparse
import os
import numpy as np
import math
import sys
import pdb

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets.mnist import MNIST
from lenet import LeNet5Half
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import resnet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'cifar10', 'cifar100'])
parser.add_argument('--data', type=str, default='/cache/data/')
parser.add_argument('--teacher_dir', type=str, default='/cache/models/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=2e-3, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=120, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--output_dir', type=str, default='/cache/models/')
parser.add_argument('--anomaly_rate', type=int, default=0.1, help='anomaly_rate')
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True

accr = 0
accr_best = 0


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(opt.channels, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(120, 16, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.deconv2 = nn.ConvTranspose2d(16, 6, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.deconv3 = nn.ConvTranspose2d(6, 1, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()

    def forward(self, feature):
        output = feature.view(-1, 120, 1, 1)
        output = self.deconv1(output)
        output = self.relu1(output)
        output = self.upsample1(output)
        output = self.deconv2(output)
        output = self.relu2(output)
        output = self.upsample2(output)
        output = self.deconv3(output)
        img = self.relu3(output)
        return img


# generator = torch.load(opt.teacher_dir + 'generator').cuda()

teacher = torch.load(opt.teacher_dir + 'teacher').cuda()
teacher.eval()
generator.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()

teacher = nn.DataParallel(teacher)
# generator = nn.DataParallel(generator)


def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) / y.shape[0]
    return l_kl


if opt.dataset == 'MNIST':
    # Configure data loader
    net = LeNet5Half().cuda()
    net = nn.DataParallel(net)
    data_test = MNIST(opt.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ]))
    data_test_loader = DataLoader(data_test, batch_size=64, shuffle=False)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
    optimizer_S = torch.optim.Adam(net.parameters(), lr=opt.lr_S)

if opt.dataset != 'MNIST':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if opt.dataset == 'cifar10':
        net = resnet.ResNet18().cuda()
        net = nn.DataParallel(net)
        data_test = CIFAR10(opt.data,
                            train=False,
                            transform=transform_test)
    if opt.dataset == 'cifar100':
        net = resnet.ResNet18(num_classes=100).cuda()
        net = nn.DataParallel(net)
        data_test = CIFAR100(opt.data,
                             train=False,
                             transform=transform_test)
    data_test_loader = DataLoader(data_test, batch_size=opt.batch_size, num_workers=0)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)

    optimizer_S = torch.optim.SGD(net.parameters(), lr=opt.lr_S, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch, learing_rate):
    if epoch < 800:
        lr = learing_rate
    elif epoch < 1600:
        lr = 0.1 * learing_rate
    else:
        lr = 0.01 * learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def multivariate_normal_distribution(x, d, mean, covariance):
    x_m = x - mean
    return (1.0 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))) * np.exp(
        -(np.matmul(np.matmul(x_m, np.linalg.inv(covariance)), x_m.reshape(-1, 1))) / 2))


def pre_calculate_possibility(pred, outputs_T):
    global flagp, zz, p
    pred = pred.cpu().detach()
    pred = pred.reshape(-1, 1)
    outputs_T = outputs_T.cpu().detach()
    # fz = z.cpu().detach
    if flagp:
        # temp_z = fz
        zz = outputs_T
        p = pred
        flagp = False
    else:
        # temp_z = np.vstack([temp_z, fz])
        zz = np.vstack([zz, outputs_T])  # (120320, 10)
        p = np.vstack([p, pred])


def data_process(data):
    global signal
    for i in range(10):
        num = len(np.where(data[:, 1] == i)[0])
        num = int(num * opt.anomaly_rate)
        if num < 1:
            break
        y = np.extract(data[:, 1] == i, data[:, 0])  # 所有标签为i的值
        y.sort()
        # print(num)
        threshold = y[num - 1]
        for s in range(znum):
            if data[s, 1] == i and data[s, 0] <= threshold:
                signal[s] = 0

    # signal = np.transpose(signal).reshape(-1, 1)


def calculate_possibility():
    global zz, znum, result, latest_var,cov_matrix
    cov_matrix = np.cov(zz, rowvar=False, bias=True)
    means = torch.mean(torch.Tensor(zz))
    for i in range(zz.shape[0]):
        result = np.append(result, multivariate_normal_distribution(zz[znum], 10, means.numpy(), cov_matrix))
        znum += 1
    result = np.transpose(result).reshape(-1, 1)
    result = np.hstack((result, p))

def save_x(x):
    global temp_x,flagx
    if flagx:
        temp_x = x
        flagx = False
    else:
        temp_x = np.vstack([temp_x, x])
# ----------
#  Training
# ----------
temp_x = []
flagx = True
# lenth = len(data_train_loader)
#第一轮
for i in range(120):
        net.train()
        delta = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
        x = generator(delta)
        save_x(x.detach().cpu())
temp_x = torch.tensor(temp_x).cuda()
signal = [1 for t in range(opt.batch_size * 120)]
flags = True
batches_done = 0
pixelwise_loss = torch.nn.L1Loss()
pixelwise_loss.cuda()
for epoch in range(opt.n_epochs):
    znum = 0
    xnum = 0
    p = []
    zz = []
    flagp = True
    flagz = True
    flagx = True
    result = []
    total_correct = 0
    avg_loss = 0.0
    if opt.dataset != 'MNIST':
        adjust_learning_rate(optimizer_S, epoch, opt.lr_S)
    if flags:
        flags = False
    else:

        for j in range(len(signal)):
            if signal[i] == 0:
                temp_x[i] = torch.zeros(1, 32, 32)
    for i in range(120):
        net.train()
        # z = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
        optimizer_G.zero_grad()
        optimizer_S.zero_grad()
        _, z = teacher(temp_x[xnum:xnum + opt.batch_size].to('cuda'), out_feature=True)
        gen_imgs = generator(z)
        outputs_T, features_T = teacher(gen_imgs, out_feature=True)

        pred = outputs_T.data.max(1)[1]
        pre_calculate_possibility(pred, outputs_T)
        loss_activation = -features_T.abs().mean()
        loss_one_hot = criterion(outputs_T, pred)
        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim=1).mean(dim=0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        loss = loss_one_hot * opt.oh + loss_information_entropy * opt.ie + loss_activation * opt.a
        # loss_xx = kdloss(outputs_T, teacher(temp_x[xnum:xnum + opt.batch_size].to('cuda'))).cuda()
        xx_pixel_loss = pixelwise_loss(temp_x[xnum:xnum + opt.batch_size].to('cuda'), gen_imgs)
        loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach())
        loss += loss_kd
        loss += xx_pixel_loss * 2e-5
        # loss += loss_xx
        loss.backward()
        xnum += opt.batch_size
        optimizer_G.step()
        optimizer_S.step()
        if i == 1:
            print("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (
            epoch, opt.n_epochs, loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(),
            loss_kd.item()))

    calculate_possibility()
    data_process(result)
    print(epoch, ':', signal.count(0))

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images = images.cuda()
            labels = labels.cuda()
            net.eval()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test_loader)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
    accr = round(float(total_correct) / len(data_test), 4)
    if accr > accr_best:
        torch.save(net,opt.output_dir + '3.29CIFAR10student')
        # torch.save(generator, opt.output_dir + '3.29CIFAR10gen')
        accr_best = accr
print(accr_best)
