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
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='/cache/data/')
parser.add_argument('--teacher_dir', type=str, default='/cache/models/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=120, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')#1
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')#5
parser.add_argument('--a', type=float, default=0.1, help='activation loss')#0.1
parser.add_argument('--pl', type=float, default=2e-5, help='pixel loss')#0.01
parser.add_argument('--output_dir', type=str, default='/cache/models/')
parser.add_argument('--anomaly_rate', type=int, default=0.1, help='anomaly_rate')#0.1
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
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.deconv1 = nn.ConvTranspose2d(120, 16, kernel_size=(5,5))
#         self.deconv2 = nn.ConvTranspose2d(16, 6, 3)
#         self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.deconv3 = nn.ConvTranspose2d(6, 3, 3)
#         self.latest_out = nn.Conv2d(3, 1, 1,stride=(1,1),padding=0)
#     def forward(self, feature):
#         z = feature.view(-1,120,1,1)
#         z = F.relu(self.deconv1(z))
#         img = F.relu(self.deconv2(z))
#         img = self.upsampling(img)
#         img = F.relu(self.deconv3(img))
#         img = self.upsampling(img)
#         img = self.latest_out(img)
#         return img
        
generator = Generator().cuda()
    
# teacher = torch.load(opt.teacher_dir + 'MNISTteacher').cuda()
teacher = torch.load(opt.teacher_dir + 'gen_test_teacher').cuda()
teacher.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()

teacher = nn.DataParallel(teacher)
generator = nn.DataParallel(generator)

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
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
    # optimizer_parameters =  itertools.chain(generator.parameters(),net.parameters())
    # optimizer_Sum = torch.optim.Adam(net.parameters(), lr=0.01)


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
# from torchvision.utils import save_image
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
        result = np.append(result, multivariate_normal_distribution(saved_output[znum], 10, 
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
# np.savetxt("C:/Users/ASUS/Desktop/temp/saved_feature.txt",np.array(saved_feature.detach().cpu()).reshape(-1, 1))
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
            saved_genimg[j] = torch.zeros(1, 32, 32)
            saved_feature[j] = torch.zeros(120)
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
        loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach()) 
        loss += loss_kd
        xx_pixel_loss = pixelwise_loss(saved_genimg[xnum:xnum + opt.batch_size], gen_imgs)
        loss += xx_pixel_loss *  opt.pl


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
        accr_best = accr
        torch.save(net,opt.output_dir + 'fanhuaxing_mine_student')
print(accr_best)
