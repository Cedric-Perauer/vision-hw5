import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import pdb
import time
import torchvision.models as torchmodels

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        if not os.path.exists('logs'):
            os.makedirs('logs')
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S_log.txt')
        self.logFile = open('logs/' + st, 'w')

    def log(self, str):
        print(str)
        self.logFile.write(str + '\n')

    def criterion(self):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001)

    def adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.lr 
        if(epoch%50 ==0):
            lr = lr * 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
       


class LazyNet(BaseModel):
    def __init__(self):
        super(LazyNet, self).__init__()
        self.lin1 = nn.Linear(32*32*3,10)



    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.lin1(x)
        return x
        

class BoringNet(BaseModel):
    def __init__(self):
        super(BoringNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)



    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x



class ResidualBlock(nn.Module): 
    def __init__(self,in_channels,out_channels,stride = 1, kernel_size = 3,padding = 1,bias = False):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),nn.BatchNorm2d(out_channels))

        if stride!=1 or in_channels !=out_channels: 
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),nn.BatchNorm2d(out_channels))
        else : 
            self.shortcut = nn.Sequential()   
    def forward(self,x): 
        residual = x #save for skip connection 
        x = self.conv1(x)
        x = self.conv2(x)
        x+= self.shortcut(residual) #add shortcut connection
        x = nn.ReLU(True)(x)
        return x 

class CoolNet(BaseModel):
    def __init__(self):
        super(CoolNet, self).__init__()
        self.block1 = nn.Sequential(
        nn.Conv2d(3,64,kernel_size=2,stride=2,padding=3,bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
       nn.MaxPool2d(1,1),
       ResidualBlock(64,64),
       ResidualBlock(64,64,2)
        )
        self.block3 = nn.Sequential(
        ResidualBlock(64,128),
        ResidualBlock(128,128,2)
        )
        self.block4 = nn.Sequential(
        ResidualBlock(128,256),
        ResidualBlock(256,256,2)
        )
        self.block5 = nn.Sequential(
        ResidualBlock(256,512),
        ResidualBlock(512,512,2)
        )
        self.avgpool = nn.AvgPool2d(2) 
        # vowel_diacritic 
        self.fc1 = nn.Linear(512,11)
        # grapheme_root
        self.fc2 = nn.Linear(512,168)
        # consonant_diacritic
        self.fc3 = nn.Linear(512,10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x3
