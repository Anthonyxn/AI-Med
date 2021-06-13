import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Conv3DTranspose
from keras.optimizers import Adam

from .metrics import dice_coefficient_loss, dice_coefficient

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def unet_model_3d(input_shape, lr=1e-5, pool_size=(2, 2, 2), n_labels=1, convtranspose3d=False, depth=4,
                  n_base_filters=32,
                  activation_name="sigmoid", metrics=dice_coefficient):
    '''

    :param input_shape:
    :param pool_size:
    :param n_labels:
    :param convtranspose3d: 是否使用Conv3DTranspose（默认UpSampling3D）
    :param depth:
    :param n_base_filters: 第一次卷积后输出维度（默认32）
    :param activation_name:
    :return:
    '''
    inputs = Input(input_shape)
    current_layer = inputs
    # layers：[encoder第i层输出]
    layers = list()

    # encoder with max_polling
    for layer_depth in range(depth):
        # 目前层第一次卷积（*2）
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters * (2 ** layer_depth))
        # 目前层第二次卷积（*2）
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters * (2 ** layer_depth) * 2)
        if layer_depth < depth - 1:
            # 此处stride先使用none，文档中要求为none或3的整数倍，但是示例代码里要求2，需要后续查证
            # encoder目前层经过max_polling后的下一层
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            layers.append(layer2)
        else:
            # 最底层，不需要进一步max_polling和储存
            current_layer = layer2
            layers.append(layer2)

    # decoder with up-convolution or up-sampling
    for layer_depth in range(depth - 2, -1, -1):
        # 从次深层开始合并
        up_convolution = get_up_convolution(pool_size=pool_size, convtranspose3d=convtranspose3d,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, layers[layer_depth]], axis=1)
        current_layer = create_convolution_block(n_filters=layers[layer_depth]._keras_shape[1],
                                                 input_layer=concat)
        current_layer = create_convolution_block(n_filters=layers[layer_depth]._keras_shape[1],
                                                 input_layer=current_layer)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)
    # 建立模型以后开始编译
    if not isinstance(metrics, list):
        metrics = [metrics]
    model.compile(optimizer=Adam(lr=lr), loss=dice_coefficient_loss, metrics=metrics)
    return model


def create_convolution_block(input_layer, n_filters, kernel=(3, 3, 3),
                             padding='same', strides=(1, 1, 1)):
    """
    conv+bn+relu
    :param input_layer:
    :param n_filters:卷积核的数目（即输出的维度）
    :param kernel:
    :param padding:补0策略
    :param strides:单个整数或由3个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides
    均与任何不为1的dilation_rate均不兼容
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    layer = BatchNormalization(axis=1)(layer)
    return Activation('relu')(layer)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       convtranspose3d=False):
    if convtranspose3d:
        return Conv3DTranspose(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)
