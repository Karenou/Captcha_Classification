import torch.nn as nn


def get_activation(name=None, alpha=None):
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leakyrelu":
        return nn.LeakyReLU(alpha)
    elif name =="sigmoid":
        return nn.Sigmoid()
    else:
        print("Please input correct activation function name")
        return None


def get_normalization(name=None, dim=None):
    if name is None:
        return None
    elif name == "batch":
        return nn.BatchNorm2d(dim)
    elif name =="instance":
        return nn.InstanceNorm2d(dim)
    else:
        print("please inout correct normalization name")
        return None


def conv_block(in_feat, out_feat, kernel_size=3, stride=1, padding=0, 
                normalize="instance", activation="relu", alpha=None):
    layers = [nn.Conv2d(in_feat, out_feat, kernel_size, stride=stride, padding=padding)]
    if normalize is not None:
        layers.append(get_normalization(name=normalize, dim=out_feat))
    if activation is not None:
        layers.append(get_activation(activation, alpha))
    return layers

def cnn_trans_block(in_feat, out_feat, kernel_size=3, stride=1, padding=0, 
                    normalize="instance", activation="relu", alpha=None):
    layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size, stride=stride, padding=padding)]
    if normalize is not None:
        layers.append(get_normalization(name=normalize, dim=out_feat))
    if activation is not None:
        layers.append(get_activation(activation, alpha))
    return layers

def fc_block(in_feat, out_feat, normalize="batch", activation="relu", alpha=None, dropout=None):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize is not None:
        layers.append(get_normalization(name=normalize, dim=out_feat))
    if dropout:
        layers.append(nn.Dropout(dropout))
    if activation is not None:
        layers.append(get_activation(activation, alpha))
    return layers


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
