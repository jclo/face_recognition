# ******************************************************************************
"""
Creates and Returns ResNet50 Convolutional Neural Network.
(referenced on https://arxiv.org/pdf/1512.03385.pdf)

This implementation is derived from the official keras implementation that
can be found here:
    . https://github.com/keras-team/keras-applications/blob/master/
      keras_applications/resnet50.py


Private Functions:
    . _conv_block                   a block that has a conv layer at shortcut,
    . _dw_conv_block                a block that has no conv layer at shortcut,


Public Functions:
    . ResNet50                      returns ResNet50 CNN,


@namespace      -
@author         Mobilabs
@since          0.0.0
@version        0.0.0
@licence        MIT. Copyright (c) 2020 Mobilabs <contact@mobilabs.fr>
"""
# ******************************************************************************
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, ZeroPadding2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization


# -- Private Functions ---------------------------------------------------------


def _conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """The conv block is the block that has a conv layer at shortcut.

    ### Parameters:
        param1 (obj):       intermediate input tensor.
        param2 (num):       the size of the kernel.
        param3 (arr):       the size of the filters.
        param4 (num):       the identity block number.
        param5 (num):       the block number inside the identity block.
        param5 (tuple):     the size of the stride.

    ### Returns:
        (obj):              returns the intermediate output tensor.

    ### Raises:
        none
    """
    filters1, filters2, filters3 = filters
    name = 'conv' + str(stage) + '_' + str(block)

    name1 = name + '_1x1_reduce'
    x = Conv2D(filters1, (1, 1), strides=strides,
               kernel_initializer='he_normal',
               use_bias=False,
               padding='same',
               name=name1)(input_tensor)
    x = BatchNormalization(name=name1 + '_bn')(x)
    x = Activation('relu', name=name1 + '_relu')(x)

    name2 = name + '_3x3'
    x = Conv2D(filters2, kernel_size,
               kernel_initializer='he_normal',
               use_bias=False,
               padding='same',
               name=name2)(x)
    x = BatchNormalization(name=name2 + '_bn')(x)
    x = Activation('relu', name=name2 + '_relu')(x)

    name3 = name + '_1x1_increase'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               use_bias=False,
               padding='same',
               name=name3)(x)
    x = BatchNormalization(name=name3 + '_bn')(x)

    name4 = name + '_1x1_proj'
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      use_bias=False,
                      padding='same',
                      name=name4)(input_tensor)
    shortcut = BatchNormalization(name=name4 + '_bn')(shortcut)

    x = keras.layers.add([x, shortcut], name=name + '_add')
    x = Activation('relu', name=name + '_add_relu')(x)
    return x


def _identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    ### Parameters:
        param1 (obj):       intermediate input tensor.
        param2 (num):       the size of the kernel.
        param3 (arr):       the size of the filters.
        param4 (num):       the identity block number.
        param5 (num):       the block number inside the identity block.

    ### Returns:
        (obj):              returns the intermediate output tensor.

    ### Raises:
        none
    """
    filters1, filters2, filters3 = filters
    name = 'conv' + str(stage) + '_' + str(block)

    name1 = name + '_1x1_reduce'
    x = Conv2D(filters1, (1, 1),
               kernel_initializer='he_normal',
               use_bias=False,
               padding='same',
               name=name1)(input_tensor)
    x = BatchNormalization(name=name1 + '_bn')(x)
    x = Activation('relu', name=name1 + '_relu')(x)

    name2 = name + '_3x3'
    x = Conv2D(filters2, kernel_size,
               kernel_initializer='he_normal',
               use_bias=False,
               padding='same',
               name=name2)(x)
    x = BatchNormalization(name=name2 + '_bn')(x)
    x = Activation('relu', name=name2 + '_relu')(x)

    name3 = name + '_1x1_increase'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               use_bias=False,
               padding='same',
               name=name3)(x)
    x = BatchNormalization(name=name3 + '_bn')(x)

    x = keras.layers.add([x, input_tensor], name=name + '_add')
    x = Activation('relu', name=name + '_add_relu')(x)
    return x


# -- Public Functions ----------------------------------------------------------

def ResNet50(input_shape=(224, 224, 3), n_classes=1000):
    """Creates the ResNet50 CNN.

    ### Parameters:
        param1 (tuple):     the shape of the input image.
        param2 (num):       the number of classes.

    ### Returns:
        (obj):              returns the Resnet50 CNN model.

    ### Raises:
        none
    """
    # Create the Tensor
    input = Input(shape=input_shape)

    # inputs are of size 224 x 224 x 3
    x = Conv2D(64, (7, 7), strides=(2, 2),
               kernel_initializer='he_normal',
               use_bias=False,
               padding='valid',
               name='conv1_7x7_s2')(input)
    x = BatchNormalization(name='conv1_7x7_s2_bn')(x)
    x = Activation('relu', name='conv1_7x7_s2_relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_7x7_s2_pad_1x1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='conv1_7x7_s2_pool_3x3')(x)

    x = _conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = _identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = _identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = _conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = _identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = _identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = _identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = _conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = _conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = _identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = _identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    # Output layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, name='predictions')(x)
    x = Activation('softmax', name='predictions_softmax')(x)

    model = keras.models.Model(input, x, name='ResNet50')
    return model


if __name__ == '__main__':
    model = ResNet50()
    model.summary()

    keras.utils.plot_model(model,
                           to_file='./diagrams/ResNet50.png',
                           show_shapes=True,
                           show_layer_names=True)

# -- o ---
