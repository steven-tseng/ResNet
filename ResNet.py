from tensorflow.keras.layers import Input, Dense, Conv2D, ReLU, Add, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from Block import convBlock1, basicConvBlock, bottleneckConvBlock

def ResNet_basicblock(imageSize, classes = 2, structure = [2,2,2,2]):
    if len(imageSize) != 3:
        raise Exception("ERROR: Wrong input shape!")

    input = Input(imageSize)
    conv1 = convBlock1(input)
    block = MaxPooling2D(pool_size=3, strides=2, padding="same")(conv1)

    # Auto construct residual blocks
    filters = 64
    for convIndex, iteration in enumerate(structure):
        for i in range(iteration):
            if i == 0 and convIndex == 0:
                block = basicConvBlock(block, filters, firstConv2Block = True, firstBlock = True)
            elif i == 0 and not convIndex == 0:
                block = basicConvBlock(block, filters, firstConv2Block = False, firstBlock = True)
            else:
                block = basicConvBlock(block, filters, firstConv2Block = False, firstBlock = False)
        filters = filters * 2

    gap = GlobalAveragePooling2D()(block)
    output = Dense(classes, activation='softmax')(gap)

    model = Model(input, output)

    return model

def ResNet_bottleneck(imageSize, classes = 2, structure = [3,4,6,3]):
    if len(imageSize) != 3:
        raise Exception("ERROR: Wrong input shape!")

    input = Input(imageSize)
    conv1 = convBlock1(input)
    block = MaxPooling2D(pool_size=3, strides=2, padding="same")(conv1)

    # Auto construct residual blocks
    filters = 64
    for convIndex, iteration in enumerate(structure):
        for i in range(iteration):
            if i == 0 and convIndex == 0:
                block = bottleneckConvBlock(block, filters, firstConv2Block = True, firstBlock = True)
            elif i == 0 and not convIndex == 0:
                block = bottleneckConvBlock(block, filters, firstConv2Block = False, firstBlock = True)
            else:
                block = bottleneckConvBlock(block, filters, firstConv2Block = False, firstBlock = False)
        filters = filters * 2

    gap = GlobalAveragePooling2D()(block)
    output = Dense(classes, activation='softmax')(gap)

    model = Model(input, output)

    return model

def ResNet18(imageSize, classes = 2):
    return ResNet_basicblock(imageSize, classes, structure = [2,2,2,2])

def ResNet34(imageSize, classes = 2):
    return ResNet_basicblock(imageSize, classes, structure = [3,4,6,3])

def ResNet50(imageSize, classes = 2):
    return ResNet_bottleneck(imageSize, classes, structure = [3,4,6,3])

def ResNet101(imageSize, classes = 2):
    return ResNet_bottleneck(imageSize, classes, structure = [3,4,23,3])

def ResNet152(imageSize, classes = 2):
    return ResNet_bottleneck(imageSize, classes, structure = [3,8,36,3])
