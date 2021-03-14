import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, ReLU, Add, BatchNormalization

def convBlock1(input, padding="same"):
    conv1 = Conv2D(64, (7,7), padding="same", strides=2)(input)
    bn1 = BatchNormalization(axis=-1)(conv1)
    relu1 = ReLU()(bn1)
    output = relu1

    return output

"""
    firstConv2Block: The first block of Conv2 whose stride is 1
    firstBlock: Except for the firstConv2Block, the stride should be 2 at the first block first layer.
"""
def basicConvBlock(input, filters, firstConv2Block = False, firstBlock = False):
    strides = 1
    if firstBlock and not firstConv2Block:
        strides = 2

    # Transform shortcut to have the same filter than output of conv layers
    shortcut = input
    conv_shortcut = Conv2D(filters, (1,1), padding="same", strides=strides)(shortcut)
    bn_shortcut = BatchNormalization(axis=-1)(conv_shortcut)

    conv1 = Conv2D(filters, (3,3), padding="same", strides=strides)(input)
    bn1 = BatchNormalization(axis=-1)(conv1)
    relu1 = ReLU()(bn1)
    conv2 = Conv2D(filters, (3,3), padding="same", strides=1)(relu1)
    bn2 = BatchNormalization(axis=-1)(conv2)

    addition1 = Add()([bn_shortcut, bn2])
    relu2 = ReLU()(addition1)

    output = relu2

    return output

def bottleneckConvBlock(input, filters, firstConv2Block = False, firstBlock = False):
    strides = 1
    if firstBlock and not firstConv2Block:
        strides = 2

    shortcut = input
    conv_shortcut = Conv2D(filters * 4, (1,1), padding="same", strides=strides)(shortcut)
    bn_shortcut = BatchNormalization(axis=-1)(conv_shortcut)

    conv1 = Conv2D(filters, (1,1), padding="same", strides=strides)(input)
    bn1 = BatchNormalization(axis=-1)(conv1)
    relu1 = ReLU()(bn1)
    conv2 = Conv2D(filters, (3,3), padding="same", strides=1)(relu1)
    bn2 = BatchNormalization(axis=-1)(conv2)
    relu2 = ReLU()(bn2)

    # For bottleneck block, the third layer has 4 times filters output eventually
    conv3 = Conv2D(filters * 4, (1,1), padding="same", strides=1)(relu2)
    bn3 = BatchNormalization(axis=-1)(conv3)
    addition1 = Add()([bn_shortcut, bn3])
    relu4 = ReLU()(addition1)

    output = relu4

    return output
