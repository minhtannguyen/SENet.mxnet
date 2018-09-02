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

def residual_unit(data, datamin, num_filter, ratio, stride, dim_match, name, num_group, bottle_neck=True,  bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNext Unit symbol for building ResNext
    Parameters
    ----------
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
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        conv_weight1 = mx.sym.Variable(name=name +'_conv1_weight', init=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
        conv1 = mx.sym.Convolution(data=data, weight=conv_weight1, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        conv1min = mx.sym.Convolution(data=datamin, weight=conv_weight1, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1min')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        bn1min = mx.sym.BatchNorm(data=conv1min, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1min')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        act1min = -mx.sym.Activation(data=-bn1min, act_type='relu', name=name + '_relu1min')
        
        conv_weight2 = mx.sym.Variable(name=name +'_conv2_weight', init=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
        conv2 = mx.sym.Convolution(data=act1, weight=conv_weight2, num_filter=int(num_filter*0.5), num_group=num_group, kernel=(3,3), stride=stride, pad=(1,1), no_bias=True, workspace=workspace, name=name + '_conv2')
        conv2min = mx.sym.Convolution(data=act1min, weight=conv_weight2, num_filter=int(num_filter*0.5), num_group=num_group, kernel=(3,3), stride=stride, pad=(1,1), no_bias=True, workspace=workspace, name=name + '_conv2min')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        bn2min = mx.sym.BatchNorm(data=conv2min, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2min')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        act2min = -mx.sym.Activation(data=-bn2min, act_type='relu', name=name + '_relu2min')
        
        conv_weight3 = mx.sym.Variable(name=name +'_conv3_weight', init=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
        conv3 = mx.sym.Convolution(data=act2, weight=conv_weight3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        conv3min = mx.sym.Convolution(data=act2min, weight=conv_weight3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3min')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        bn3min = mx.sym.BatchNorm(data=conv3min, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3min')

        squeeze = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_squeeze')
        squeezemin = mx.sym.Pooling(data=bn3min, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_squeezemin')
        squeeze = mx.symbol.Flatten(data=squeeze, name=name + '_flatten')
        squeezemin = mx.symbol.Flatten(data=squeezemin, name=name + '_flattenmin')
        
        fc_weight1 = mx.sym.Variable(name=name +'_excitation1_weight', init=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
        fc_bias1 = mx.sym.Variable(name=name +'_excitation1_bias', init=mx.init.Zero())
        excitation = mx.symbol.FullyConnected(data=squeeze, weight=fc_weight1, bias=fc_bias1, num_hidden=int(num_filter*ratio), name=name + '_excitation1')
        excitationmin = mx.symbol.FullyConnected(data=squeezemin, weight=fc_weight1, bias=fc_bias1, num_hidden=int(num_filter*ratio), name=name + '_excitation1min')
        excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + '_excitation1_relu')
        excitationmin = -mx.sym.Activation(data=-excitationmin, act_type='relu', name=name + '_excitation1_relumin')
        
        fc_weight2 = mx.sym.Variable(name=name +'_excitation2_weight', init=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
        fc_bias2 = mx.sym.Variable(name=name +'_excitation2_bias', init=mx.init.Zero())
        excitation = mx.symbol.FullyConnected(data=excitation, weight=fc_weight2, bias=fc_bias2, num_hidden=num_filter, name=name + '_excitation2')
        excitationmin = mx.symbol.FullyConnected(data=excitationmin, weight=fc_weight2, bias=fc_bias2, num_hidden=num_filter, name=name + '_excitation2min')
        excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + '_excitation2_sigmoid')
        excitationmin = -mx.sym.Activation(data=-excitationmin, act_type='sigmoid', name=name + '_excitation2_sigmoidmin')
        bn3 = mx.symbol.broadcast_mul(bn3, mx.symbol.reshape(data=excitation, shape=(-1, num_filter, 1, 1)))
        bn3min = mx.symbol.broadcast_mul(bn3min, mx.symbol.reshape(data=excitationmin, shape=(-1, num_filter, 1, 1)))

        if dim_match:
            shortcut = data
            shortcutmin = datamin
        else:
            shortcut_weight1 = mx.sym.Variable(name=name + '_sc_weight', init=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
            shortcut_conv = mx.sym.Convolution(data=data, weight=shortcut_weight1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
            shortcut_convmin = mx.sym.Convolution(data=datamin, weight=shortcut_weight1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_scmin')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')
            shortcutmin = mx.sym.BatchNorm(data=shortcut_convmin, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bnmin')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
            shortcutmin._set_attr(mirror_stage='True')
        eltwise =  bn3 + shortcut
        eltwisemin =  bn3min + shortcutmin
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu'), -mx.sym.Activation(data=-eltwise, act_type='relu', name=name + '_relumin')
    else:
        conv_weight1 = mx.sym.Variable(name=name + '_conv_weight1', init=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
        conv1 = mx.sym.Convolution(data=data, weight=conv_weight1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        conv1min = mx.sym.Convolution(data=datamin, weight=conv_weight1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1min')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        bn1min = mx.sym.BatchNorm(data=conv1min, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1min')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        act1min = -mx.sym.Activation(data=-bn1min, act_type='relu', name=name + '_relu1min')
        
        conv_weight2 = mx.sym.Variable(name=name + '_conv_weight2', init=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
        conv2 = mx.sym.Convolution(data=act1, weight=conv_weight2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        conv2min = mx.sym.Convolution(data=act1min, weight=conv_weight2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2min')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        bn2min = mx.sym.BatchNorm(data=conv2min, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2min')

        squeeze = mx.sym.Pooling(data=bn2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_squeeze')
        squeezemin = mx.sym.Pooling(data=bn2min, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_squeezemin')
        squeeze = mx.symbol.Flatten(data=squeeze, name=name + '_flatten')
        squeezemin = mx.symbol.Flatten(data=squeezemin, name=name + '_flattenmin')
        
        fc_weight1 = mx.sym.Variable(name=name +'_fc_weight1', init=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
        fc_bias1 = mx.sym.Variable(name=name + '_fc_bias1', init=mx.init.Zero())
        excitation = mx.symbol.FullyConnected(data=squeeze, weight=fc_weight1, bias=fc_bias1, num_hidden=int(num_filter*ratio), name=name + '_excitation1')
        excitationmin = mx.symbol.FullyConnected(data=squeezemin, weight=fc_weight1, bias=fc_bias1, num_hidden=int(num_filter*ratio), name=name + '_excitation1min')
        excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + '_excitation1_relu')
        excitationmin = -mx.sym.Activation(data=-excitationmin, act_type='relu', name=name + '_excitation1_relumin')
        
        fc_weight2 = mx.sym.Variable(name=name + '_fc_weight2', init=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
        fc_bias2 = mx.sym.Variable(name=name + '_fc_bias2', init=mx.init.Zero())
        excitation = mx.symbol.FullyConnected(data=excitation, weight=fc_weight2, bias=fc_bias2, num_hidden=num_filter, name=name + '_excitation2')
        excitationmin = mx.symbol.FullyConnected(data=excitationmin, weight=fc_weight2, bias=fc_bias2, num_hidden=num_filter, name=name + '_excitation2min')
        excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + '_excitation2_sigmoid')
        excitationmin = -mx.sym.Activation(data=-excitationmin, act_type='sigmoid', name=name + '_excitation2_sigmoidmin')
        bn2 = mx.symbol.broadcast_mul(bn2, mx.symbol.reshape(data=excitation, shape=(-1, num_filter, 1, 1)))
        bn2min = mx.symbol.broadcast_mul(bn2min, mx.symbol.reshape(data=excitationmin, shape=(-1, num_filter, 1, 1)))

        if dim_match:
            shortcut = data
            shortcutmin = datamin
        else:
            shortcut_weight1 = mx.sym.Variable(name=name + '_shortcut_weight1', init=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
            shortcut_conv = mx.sym.Convolution(data=data, weight=shortcut_weight1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
            shortcut_convmin = mx.sym.Convolution(data=datamin, weight=shortcut_weight1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_scmin')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')
            shortcutmin = mx.sym.BatchNorm(data=shortcut_convmin, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bnmin')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
            shortcutmin._set_attr(mirror_stage='True')
        eltwise = bn2 + shortcut
        eltwisemin = bn2min + shortcutmin
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu'), -mx.sym.Activation(data=-eltwisemin, act_type='relu', name=name + '_relumin')
		
def resnext(units, num_stage, filter_list, ratio_list, num_class, num_group, data_type, drop_out, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
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
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable(name='softmax_label') 
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    elif data_type == 'imagenet':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        # min branch
        bodymin = -body
        bodymin = mx.sym.Activation(data=bodymin, act_type='relu', name='relu0')
        
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
        # min branch
        bodymin = mx.symbol.Pooling(data=bodymin, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
        bodymin = -bodymin
        
    elif data_type == 'vggface':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        # min branch
        bodymin = -body
        bodymin = mx.sym.Activation(data=bodymin, act_type='relu', name='relu0')
        
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
        # min branch
        bodymin = mx.symbol.Pooling(data=bodymin, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
        bodymin = -bodymin
        
    elif data_type == 'msface':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        # min branch
        bodymin = -body
        bodymin = mx.sym.Activation(data=bodymin, act_type='relu', name='relu0')
        
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
        # min branch
        bodymin = mx.symbol.Pooling(data=bodymin, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
        bodymin = -bodymin
        
    else:
         raise ValueError("do not support {} yet".format(data_type))
    for i in range(num_stage):
        body, bodymin = residual_unit(body, bodymin, filter_list[i+1], ratio_list[2], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), num_group=num_group, bottle_neck=bottle_neck,  
                             bn_mom=bn_mom, workspace=workspace, memonger=memonger)
        for j in range(units[i]-1):
            body, bodymin = residual_unit(body, bodymin, filter_list[i+1], ratio_list[2], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 num_group=num_group, bottle_neck=bottle_neck, bn_mom=bn_mom, workspace=workspace, memonger=memonger)        
    pool1 = mx.symbol.Pooling(data=body, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    pool1min = mx.symbol.Pooling(data=bodymin, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1min')
    flat = mx.symbol.Flatten(data=pool1)
    flatmin = mx.symbol.Flatten(data=pool1min)
    drop1= mx.symbol.Dropout(data=flat, p=drop_out, name='dp1')
    drop1min= mx.symbol.Dropout(data=flatmin, p=drop_out, name='dp1min')
    
    fc_weight_sf = mx.sym.Variable(name='fc1_weight', init=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
    fc_bias_sf = mx.sym.Variable(name='fc1_bias', init=mx.init.Zero())
    fc1 = mx.symbol.FullyConnected(data=drop1, weight=fc_weight_sf, bias=fc_bias_sf, num_hidden=num_class, name='fc1')
    fc1min = mx.symbol.FullyConnected(data=drop1min, weight=fc_weight_sf, bias=fc_bias_sf, num_hidden=num_class, name='fc1min')
    
    sf1 = mx.symbol.SoftmaxOutput(data=fc1, label=label, grad_scale=0.5,  name='softmax')
    sf1min = mx.symbol.SoftmaxOutput(data=fc1min, label=label, grad_scale=0.5,  name='softmaxmin')
    sf = 0.5*(sf1 + sf1min)
    return sf
    