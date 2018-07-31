# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 09:13:59 2018

@author: user
"""

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import cuda

xp = cuda.cupy

# Define Recurrent Block
class Block(chainer.Chain):
    """
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter in ffconv & fbconv is ksize x ksize.
        pad (int): The padding to use for the convolution.
        LoopTimes (int): The number of recurrent cycles.
    """

    def __init__(self, in_channels, out_channels, LoopTimes=5):
        super(Block, self).__init__()
        with self.init_scope():
            """
            ffconv : feedforword Convolution.
            fbconv : feedback Convolution (deconvolution).
            bpconv : bypass Convolution (1x1 convolution).
            update_late : A learnable and non-negative parameter.
            """
            self.bn = L.BatchNormalization(in_channels)
            self.ffconv = L.Convolution2D(in_channels, out_channels, ksize=3, pad=1)
            self.fbconv = L.Deconvolution2D(out_channels, in_channels, ksize=3, pad=1)
            self.bpconv = L.Convolution2D(in_channels, out_channels, ksize=1)
            self.update_rate = L.Scale(axis=0, W_shape=(1,))

            self.LoopTimes = LoopTimes
            self.out_channels = out_channels

    def __call__(self, r):
        # Define initial input
        rbn = self.bn(r) #Check shape with print(r.shape)
        r0 = F.relu(self.ffconv(rbn))

        # Set update rate
        one = Variable(xp.array([1],  dtype=xp.float32))
        update_rate = F.absolute(self.update_rate(one))

        # Recurrent loop
        for t in range(self.LoopTimes):
            if t == 0:
                rt = r0
            pt = self.fbconv(rt) #Check shape with print(pt.shape)
            e = F.relu(r - pt) # r & pt shape is the same.
            rt = rt + F.scale(self.ffconv(e), update_rate)
        return rt + self.bpconv(rbn)

# Define Local Predictive Coding Network
class LocalPCN(chainer.Chain):
    def __init__(self, class_labels=10, LoopTimes=5, return_out=False):
        super(LocalPCN, self).__init__(
                block1 = Block(3, 64, LoopTimes=LoopTimes),
                block2 = Block(64, 64, LoopTimes=LoopTimes),
                block3 = Block(64, 128, LoopTimes=LoopTimes),
                block4 = Block(128, 128, LoopTimes=LoopTimes),
                block5 = Block(128, 256, LoopTimes=LoopTimes),
                block6 = Block(256, 256, LoopTimes=LoopTimes),
                block7 = Block(256, 512, LoopTimes=LoopTimes),
                block8 = Block(512, 512, LoopTimes=LoopTimes),
                fc = L.Linear(512, class_labels, nobias=True)
                )
        self.return_out = return_out

    def __call__(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.block4(h)
        h = self.block5(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.block6(h)
        h = self.block7(h)
        h = self.block8(h)
        h = F.average(h, axis=(2,3)) # Global Average Pooling
        return self.fc(h)
