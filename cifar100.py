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
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import resnet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar100', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='/cache/data/')
parser.add_argument('--teacher_dir', type=str, default='/cache/models/')
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.002, help='learning rate')
parser.add_argument('--lr_S', type=float, default=0.1, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=512, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--oh', type=float, default=0.5, help='one hot loss')
parser.add_argument('--ie', type=float, default=20, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--output_dir', type=str, default='/cache/models/')
parser.add_argument('--anomaly_rate', type=int, default=0.05, help='anomaly_rate')#0.1

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
        
generator = Generator().cuda()
    
teacher = torch.load(opt.teacher_dir +'teacher-CIFAR100').cuda()
teacher.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()

teacher = nn.DataParallel(teacher)
generator = nn.DataParallel(generator)

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl


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
        lr = 0.1*learing_rate
    else:
        lr = 0.01*learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def saving_feature(x):
    global saved_feature,flagF
    if flagF:
        saved_feature = x
        flagF = False
    else:
        saved_feature = np.vstack([saved_feature, x])
   
def saving_feature2(x):
    global saved_feature2,flagF2
    if flagF2:
        saved_feature2 = x
        flagF2 = False
    else:
        saved_feature2 = np.vstack([saved_feature2, x])

def saving_genimg(x):
    global saved_genimg,flagG
    if flagG:
        saved_genimg = x
        flagG = False
    else:
        saved_genimg = np.vstack([saved_genimg, x])

def saving_genimg2(x):
    global saved_genimg2,flagG2
    if flagG2:
        saved_genimg2 = x
        flagG2 = False
    else:
        saved_genimg2 = np.vstack([saved_genimg2, x])
pixelwise_loss = torch.nn.L1Loss()
pixelwise_loss.cuda()
def pre_calculate_possibility(pred, outputs_T):
    global flagp, saved_output, p
    pred = pred.cpu().detach()
    pred = pred.reshape(-1, 1)
    outputs_T = outputs_T.cpu().detach()
    # fz = z.cpu().detach
    if flagp:
        # temp_z = fz
        saved_output = outputs_T
        p = pred
        flagp = False
    else:
        # temp_z = np.vstack([temp_z, fz])
        saved_output = np.vstack([saved_output, outputs_T])  # (120320, 10)
        p = np.vstack([p, pred])

def multivariate_normal_distribution(x, d, mean, covariance):
    x_m = x - mean
    return (1.0 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))) * np.exp(
        -(np.matmul(np.matmul(x_m, np.linalg.inv(covariance)), x_m.reshape(-1, 1))) / 2))


def calculate_possibility():
    global saved_output, result, znum
    cov_matrix = np.cov(saved_output, rowvar=False, bias=True)
    means = torch.mean(torch.Tensor(saved_output))
    for i in range(saved_output.shape[0]):
        result = np.append(result, multivariate_normal_distribution(saved_output[znum], 100, 
                                                                    means.numpy(), cov_matrix))
        znum += 1
    result = np.transpose(result).reshape(-1, 1)
    result = np.hstack((result, p))
    
def data_process(data):
    global signal,znum
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

# ----------
#  Training
# ----------


for epoch in range(opt.n_epochs):
    znum = 0
    saved_genimg2 = []
    flagG2 = True
    saved_feature2 = []
    flagF2 = True
    total_correct = 0
    avg_loss = 0.0
    xnum = 0
    saved_feature = []
    flagF = True
    saved_genimg = []
    flagG = True
    flagp = True
    saved_output = []
    p = []
    result = []
    if opt.dataset != 'MNIST':
        adjust_learning_rate(optimizer_S, epoch, opt.lr_S)

    for i in range(120):
        net.train()
        z = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
        optimizer_G.zero_grad()
        optimizer_S.zero_grad()        
        gen_imgs = generator(z)
        saving_genimg(gen_imgs.detach().cpu().numpy())
        outputs_T, features_T = teacher(gen_imgs, out_feature=True)   
        saving_feature(features_T.detach().cpu().numpy())
        pred = outputs_T.data.max(1)[1]
        pre_calculate_possibility(pred, outputs_T)
        loss_activation = -features_T.abs().mean()
        loss_one_hot = criterion(outputs_T,pred)
        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        loss = loss_one_hot * opt.oh + loss_information_entropy * opt.ie + loss_activation * opt.a
        loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach()) 
        loss += loss_kd
        loss.backward()
        optimizer_G.step()
        optimizer_S.step() 
    calculate_possibility()
    signal = [1 for t in range(znum)]
    data_process(result)
    
    for j in range(len(signal)):
        if signal[j] == 0:
            saved_genimg[j] = torch.zeros(opt.channels, 32, 32)
            saved_feature[j] = torch.zeros(opt.latent_dim)
    saved_feature = torch.Tensor(saved_feature).cuda()
    saved_genimg = torch.Tensor(saved_genimg).cuda()
    for i in range(120):
        net.train()
        z = saved_feature[xnum:xnum + opt.batch_size]
        optimizer_G.zero_grad()
        optimizer_S.zero_grad()        
        gen_imgs = generator(z)
        # saving_genimg2(gen_imgs.detach().cpu().numpy())
        outputs_T, features_T = teacher(gen_imgs, out_feature=True)  
        # saving_feature2(features_T.detach().cpu().numpy())
        pred = outputs_T.data.max(1)[1]       
        loss_activation = -features_T.abs().mean()
        loss_one_hot = criterion(outputs_T,pred)
        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        loss = loss_one_hot * opt.oh + loss_information_entropy * opt.ie + loss_activation * opt.a
        outputs_S, features_S = net(gen_imgs.detach(), out_feature=True)
        loss_kd = kdloss(outputs_S, outputs_T.detach())
        loss += loss_kd
        xx_pixel_loss = pixelwise_loss(saved_genimg[xnum:xnum + opt.batch_size], gen_imgs)
        loss += xx_pixel_loss * (2e-3)
        loss.backward()
        optimizer_G.step()
        optimizer_S.step()
        xnum += opt.batch_size
        if i == 1:
            print ("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, opt.n_epochs,loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item()))
            print('reconstruction error:',xx_pixel_loss)
            
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
        # torch.save(net,opt.output_dir + 'student')
        accr_best = accr
        print("Best Acc=%.8f" % accr_best)
print("Best Acc=%.8f"%accr_best)
