'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py 
Original author Wei Wu

Implemented the following paper:
Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He. "Aggregated Residual Transformations for Deep Neural Network" CVPR 2017 https://arxiv.org/pdf/1611.05431v2.pdf
Jie Hu, Li Shen, Gang Sun. "Squeeze-and-Excitation Networks" https://arxiv.org/pdf/1709.01507v1.pdf

This modification version is based on ResNet v1
This modificaiton version adds dropout layer followed by last pooling layer.
Modified by Lin Xiong Feb-11, 2017
Updated by Lin Xiong Jul-21, 2017
Added Squeeze-and-Excitation block by Lin Xiong Sep-13, 2017
'''
import mxnet as mx

from mxnet.gluon import nn
from mxnet import nd

import numpy as np

class NReLu(nn.HybridBlock):
    """
    -max(-x,0)
    Parameters
    ----------
    Input shape: (N, C, W, H)
    Output shape: (N, C * W * H)
    """
    def __init__(self, **kwargs):
        super(NReLu, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return -F.Activation(-x, act_type='relu')

class residual_unit(nn.HybridBlock):
    """Return ResNext Unit symbol for building ResNext
    Parameters
    ----------gl
    data : str
        Input data
    num_filter : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    bottle_neck : Boolen
        Whether or not to adopt bottle_neck trick as did in ResNet
    num_group : int
        Number of convolution groupes
    bn_mom : float
        Momentum of batch normalization
    workspace : int
        Workspace used in convolution operator
    """
    def __init__(self, in_channels, num_filter, ratio, strides, dim_match, name, num_group,  bn_mom=0.9, **kwargs):
        super(residual_unit, self).__init__(**kwargs)
        self.dim_match = dim_match
        self.num_filter = num_filter
        # block 1
        self.conv1 = nn.Conv2D(in_channels=in_channels, channels=int(num_filter*0.5), kernel_size=(1,1), strides=(1,1), padding=(0,0), use_bias=False, prefix=name + '_conv1_')
        self.bn1 = nn.BatchNorm(in_channels=int(num_filter*0.5), epsilon=2e-5, momentum=bn_mom, prefix=name + '_bn1_')
        self.bn1min = nn.BatchNorm(in_channels=int(num_filter*0.5), epsilon=2e-5, momentum=bn_mom, prefix=name + '_bn1min_')
        self.relu1 = nn.Activation(activation='relu', prefix=name + '_relu1_')
        self.relu1min = NReLu(prefix=name + '_relu1min_')
        
        # block 2
        self.conv2 = nn.Conv2D(in_channels=int(num_filter*0.5), channels=int(num_filter*0.5), groups=num_group, kernel_size=(3,3), strides=strides, padding=(1,1), use_bias=False, prefix=name + '_conv2_')
        self.bn2 = nn.BatchNorm(in_channels=int(num_filter*0.5), epsilon=2e-5, momentum=bn_mom, prefix=name + '_bn2_')
        self.bn2min = nn.BatchNorm(in_channels=int(num_filter*0.5), epsilon=2e-5, momentum=bn_mom, prefix=name + '_bn2min_')
        self.relu2 = nn.Activation(activation='relu', prefix=name + '_relu2_')
        self.relu2min = NReLu(prefix=name + '_relu2min_')
        
        # block 3
        self.conv3 = nn.Conv2D(in_channels=int(num_filter*0.5), channels=num_filter, kernel_size=(1,1), strides=(1,1), padding=(0,0), use_bias=False, prefix=name + '_conv3_')
        self.bn3 = nn.BatchNorm(in_channels=num_filter, epsilon=2e-5, momentum=bn_mom, prefix=name + '_bn3_')
        self.bn3min = nn.BatchNorm(in_channels=num_filter, epsilon=2e-5, momentum=bn_mom, prefix=name + '_bn3min_')
        
        # squeeze
        self.pool = nn.GlobalAvgPool2D(prefix=name + '_squeeze_') 
        self.flatten = nn.Flatten(prefix=name + '_flatten_')
        
        # excitation 1
        self.fc1 = nn.Dense(units=int(num_filter*ratio), in_units=num_filter, prefix=name + '_excitation1_')
        self.reluex1 = nn.Activation(activation='relu', prefix=name + '_excitation1_relu_')
        self.reluex1min = NReLu(prefix=name + '_excitation1_relumin_')
        
        # excitation 2
        self.fc2 = nn.Dense(units=num_filter, in_units=int(num_filter*ratio), prefix=name + '_excitation2_')
        self.reluex2 = nn.Activation(activation='relu', prefix=name + '_excitation2_relu_')
        self.reluex2min = NReLu(prefix=name + '_excitation2_relumin_')
        
        if not dim_match:
            self.fc_sc = nn.Conv2D(in_channels=in_channels, channels=num_filter, kernel_size=(1,1), strides=strides, use_bias=False, prefix=name + '_sc_')
            self.bn_sc = nn.BatchNorm(in_channels=num_filter, epsilon=2e-5, momentum=bn_mom, prefix=name + '_sc_bn_')
            self.bn_scmin = nn.BatchNorm(in_channels=num_filter, epsilon=2e-5, momentum=bn_mom, prefix=name + '_sc_bnmin_')
            
        self.relu3 = nn.Activation(activation='relu', prefix=name + '_relu3_')
        self.relu3min = NReLu(prefix=name + '_relu3min_')
        
    def hybrid_forward(self, F, x):
        xmax = x[0]; xmin = x[1]
        
        # block 1
        xmax = self.conv1(xmax)
        xmax = self.bn1(xmax)
        xmax = self.relu1(xmax)
        
        xmin = self.conv1(xmin)
        xmin = self.bn1min(xmin)
        xmin = self.relu1min(xmin)
        
        # block 2
        xmax = self.conv2(xmax)
        xmax = self.bn2(xmax)
        xmax = self.relu2(xmax)
        
        xmin = self.conv2(xmin)
        xmin = self.bn2min(xmin)
        xmin = self.relu2min(xmin)
        
        # block 3
        xmax = self.conv3(xmax)
        bn3max = self.bn3(xmax)
        
        xmin = self.conv3(xmin)
        bn3min = self.bn3min(xmin)
        
        # squeeze
        xmax = self.pool(bn3max)
        xmax = self.flatten(xmax)
        
        xmin = self.pool(bn3min)
        xmin = self.flatten(xmin)
        
        # excitation 1
        xmax = self.fc1(xmax)
        xmax = self.reluex1(xmax)
        
        xmin = self.fc1(xmin)
        xmin = self.reluex1min(xmin)
        
        # excitation 2
        xmax = self.fc2(xmax)
        xmax = self.reluex2(xmax)
        
        xmin = self.fc2(xmin)
        xmin = self.reluex2min(xmin)
        
        bn3 = F.broadcast_mul(bn3, F.reshape(data=xmax, shape=(-1, self.num_filter, 1, 1)))
        bn3min = F.broadcast_mul(bn3min, F.reshape(data=xmin, shape=(-1, self.num_filter, 1, 1)))
        
        if self.dim_match:
            shortcut = x[0]
            shortcutmin = x[1]
        else:
            shortcut = self.fc_sc(x[0])
            shortcut = self.bn_sc(shorcut)
            
            shortcutmin = self.fc_sc(x[1])
            shortcutmin = self.bn_scmin(shorcutmin)
        
        xmax = bn3 + shortcut
        xmin = bn3min + shortcutmin
        
        xmax = self.relu3(xmax)
        xmin = self.relu3min(xmin)
        
        return xmax, xmin
		
class se_resnext(nn.HybridBlock):
    """Return ResNeXt symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    num_groupes: int
		Number of convolution groups
    drop_out : float
        Probability of an element to be zeroed. Default = 0.0
    data_type : str
        Dataset type, only cifar10, imagenet and vggface supports
    workspace : int
        Workspace used in convolution operator
    """
    def __init__(self, units, num_stage, filter_list, ratio_list, num_class, num_group, data_type, drop_out, bn_mom=0.9, **kwargs):
        super(se_resnext, self).__init__(**kwargs)
        num_unit = len(units)
        assert(num_unit == num_stage)
        
        self.conv0 = nn.Conv2D(in_channels=3, channels=filter_list[0], kernel_size=(7,7), strides=(2,2), padding=(3,3), use_bias=False, prefix='conv0_')
        self.bn0 = nn.BatchNorm(in_channels=filter_list[0], epsilon=2e-5, momentum=bn_mom, prefix='bn0_')
        self.relu0 = nn.Activation(activation='relu', prefix='relu0_')
        self.relu0min = NReLu(prefix='relu0min_')
        self.pool0 = nn.MaxPool2D(pool_size=(3,3), strides=(2,2), padding=(1,1), prefix='pool0_')
        
        self.residual_stages = nn.HybridSequential(prefix='residual_')
        for i in range(num_stage):
            self.residual_stages.add(residual_unit(in_channels=filter_list[i], num_filter=filter_list[i+1], ratio=ratio_list[2], strides=(1 if i==0 else 2, 1 if i==0 else 2), dim_match=False, name='stage%d_unit%d' % (i + 1, 1), num_group=num_group, bn_mom=bn_mom, prefix='stage%d_unit%d_' % (i + 1, 1)))
            for j in range(units[i]-1):
                self.residual_stages.add(residual_unit(in_channels=filter_list[i+1], num_filter=filter_list[i+1], ratio=ratio_list[2], strides=(1,1), dim_match=True, name='stage%d_unit%d' % (i + 1, j + 2), num_group=num_group, bn_mom=bn_mom, prefix='stage%d_unit%d_' % (i + 1, j + 2)))
                
        self.pool1 = nn.GlobalAvgPool2D(prefix='pool1_')
        self.flatten1 = nn.Flatten(prefix='flatten1_')
        self.drop1 = nn.Dropout(rate=drop_out, prefix='dp1_')
        
        self.fc = nn.Dense(units=num_class, in_units=filter_list[-1], prefix='fc_')
        
    def hybrid_forward(self, F, x):
        x = self.conv0(x)
        x = self.bn0(x)
        xmax = self.relu0(x)
        xmin = self.relu0min(x)
        xmax = self.pool0(xmax)
        xmin = -self.pool0(-xmin)
        
        xmax, xmin = self.residual_stages([xmax, xmin])
        
        xmax = self.pool1(xmax)
        xmax = self.flatten1(xmax)
        xmax = self.drop1(xmax)
        
        xmin = self.pool1(xmin)
        xmin = self.flatten1(xmin)
        xmin = self.drop1(xmin)
        
        xmax = self.fc(xmax)
        xmin = self.fc(xmin)
        
        return xmax, xmin
    