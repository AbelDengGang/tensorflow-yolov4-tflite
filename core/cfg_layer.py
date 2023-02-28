# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf

# import common as common

# From Darknet
_LEAKY_RELU_ALPHA = 0.1
_BATCH_NORM_MOMENTUM = 0.9
_BATCH_NORM_EPSILON = 1e-05


_activation_dict = {
    'leaky': lambda x, y: tf.nn.leaky_relu(x, name=y, alpha=_LEAKY_RELU_ALPHA),
    'relu': lambda x, y: tf.nn.relu(x, name=y)
}


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output

# def block_tiny(input_layer, input_channel, filter_num1, activate_type='leaky'):
#     conv = convolutional(input_layer, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#     short_cut = input_layer
#     conv = convolutional(conv, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#
#     input_data = tf.concat([conv, short_cut], axis=-1)
#     return residual_output

def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')



def none_to_m1(x):
    return x if x is not None else -1


def old_reorg_layer(net, stride=2, name='reorg'):
    batch_size, height, width, channels = net.get_shape().as_list()
    batch_size = none_to_m1(batch_size)
    _height, _width, _channel = height // stride, width // stride, channels * stride * stride
    with tf.name_scope(name):
        net = tf.reshape(net, [batch_size, _height, stride, _width, stride, channels])
        net = tf.transpose(net, [0, 1, 3, 2, 4, 5])  # batch_size, _height, _width, stride, stride, channels
        net = tf.reshape(net, [batch_size, _height, _width, stride * stride * channels], name='reorg')
    return net


def reorg_layer(net, stride=2, name='reorg'):
    with tf.name_scope(name):
        net = tf.extract_image_patches(net,
                                       ksizes=[1, stride, stride, 1],
                                       strides=[1, stride, stride, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')
    return net



# # Syntax
# def cfg_layerName(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
#     pass


def cfg_net(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    width = int(param["width"])
    height = int(param["height"])
    channels = int(param["channels"])
    net = tf.keras.layers.Input([width, width, channels])
    return net



def cfg_convolutional(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    groups = int(param.get('groups',1))
    if groups > 1:
        pass
        # return cfg_group_convolutional(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose)
    del groups
    
    batch_normalize = 'batch_normalize' in param
    size = int(param['size'])
    filters = int(param['filters'])
    stride = int(param['stride'])
    pad = 'same' if param['pad'] == '1' else 'valid'
    activation = None
    weight_size = C * filters * size * size
    if "activation" in param:
        activation = _activation_dict.get(param['activation'], None)



    biases, scales, rolling_mean, rolling_variance, weights = \
        weights_walker.get_weight(param['name'],
                                  filters=filters,
                                  weight_size=weight_size,
                                  batch_normalize=batch_normalize)
    # tensor shape in tensorflow is [size,size,channel,filters]
    weights = weights.reshape(filters, C, size, size).transpose([2, 3, 1, 0])
    # print(weights.shape)
    conv_args = {
        "filters": filters,
        "kernel_size": size,
        "strides": stride,
        "activation": None,
        "padding": pad
    }

    if const_inits:
        conv_args.update({
            "kernel_initializer": tf.initializers.constant(weights),
            "bias_initializer": tf.initializers.constant(biases)
        })

    if batch_normalize:
        conv_args.update({
            "use_bias": False
        })

    net = tf.keras.layers.Conv2D(name=scope, **conv_args)(net)

    if batch_normalize:
        batch_norm_args = {
            "momentum": _BATCH_NORM_MOMENTUM,
            "epsilon": _BATCH_NORM_EPSILON,
            "fused": True,
            "trainable": training,
            # "training": training
        }

        if const_inits:
            batch_norm_args.update({
                "beta_initializer": tf.initializers.constant(biases),
                "gamma_initializer": tf.initializers.constant(scales),
                "moving_mean_initializer": tf.initializers.constant(rolling_mean),
                "moving_variance_initializer": tf.initializers.constant(rolling_variance)
            })

        net = tf.keras.layers.BatchNormalization(name=scope+'/BatchNorm', **batch_norm_args)(net,training)

    if activation:
        net = activation(net, scope+'/Activation')

    return net



def cfg_convolutional_org(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    groups = int(param.get('groups',1))
    if groups > 1:
        pass
        # return cfg_group_convolutional(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose)
    del groups
    
    batch_normalize = 'batch_normalize' in param
    size = int(param['size'])
    filters = int(param['filters'])
    stride = int(param['stride'])
    pad = 'same' if param['pad'] == '1' else 'valid'
    activation = None
    weight_size = C * filters * size * size
    if "activation" in param:
        activation = _activation_dict.get(param['activation'], None)

    biases, scales, rolling_mean, rolling_variance, weights = \
        weights_walker.get_weight(param['name'],
                                  filters=filters,
                                  weight_size=weight_size,
                                  batch_normalize=batch_normalize)
    weights = weights.reshape(filters, C, size, size).transpose([2, 3, 1, 0])

    conv_args = {
        "filters": filters,
        "kernel_size": size,
        "strides": stride,
        "activation": None,
        "padding": pad
    }

    if const_inits:
        conv_args.update({
            "kernel_initializer": tf.initializers.constant(weights, verify_shape=True),
            "bias_initializer": tf.initializers.constant(biases, verify_shape=True)
        })

    if batch_normalize:
        conv_args.update({
            "use_bias": False
        })

    net = tf.layers.conv2d(net, name=scope, **conv_args)

    if batch_normalize:
        batch_norm_args = {
            "momentum": _BATCH_NORM_MOMENTUM,
            "epsilon": _BATCH_NORM_EPSILON,
            "fused": True,
            "trainable": training,
            "training": training
        }

        if const_inits:
            batch_norm_args.update({
                "beta_initializer": tf.initializers.constant(biases, verify_shape=True),
                "gamma_initializer": tf.initializers.constant(scales, verify_shape=True),
                "moving_mean_initializer": tf.initializers.constant(rolling_mean, verify_shape=True),
                "moving_variance_initializer": tf.initializers.constant(rolling_variance, verify_shape=True)
            })

        net = tf.layers.batch_normalization(net, name=scope+'/BatchNorm', **batch_norm_args)

    if activation:
        net = activation(net, scope+'/Activation')

    return net



def cfg_dropout(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    if training:
        net = tf.layers.dropout(net,rate=param['probability'],  training=training)
    return net


def cfg_maxpool(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):

    ksize = int(param['size'])
    stride = int(param['stride'])
    net = tf.nn.max_pool(net, ksize=ksize, padding='SAME', strides=stride)

    return net



def cfg_maxpool_org(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    pool_args = {
        "pool_size": int(param['size']),
        "strides": int(param['stride']),
        "padding": 'same'
    }

    net = tf.layers.max_pooling2d(net, name=scope, **pool_args)
    return net


def cfg_avgpool(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    # Darknet uses only global avgpool (no stride, kernel size == input size)
    # Reference:
    # https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/avgpool_layer.c#L7
    assert len(param) == 1, "Expected global avgpool; no stride / size param but got param=%s" % param
    pool_args = {
        "pool_size": (H, W),
        "strides": 1
    }

    net = tf.layers.average_pooling2d(net, name=scope, **pool_args)
    return net


def cfg_route(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    if not isinstance(param["layers"], list):
        param["layers"] = [param["layers"]]
    net_index = [int(x) for x in param["layers"]]
    nets = [stack[x] for x in net_index]

    net = tf.concat(nets, axis=-1, name=scope)
    return net


def cfg_reorg(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    reorg_args = {
        "stride": int(param['stride'])
    }

    net = reorg_layer(net, name=scope, **reorg_args)
    return net


def cfg_shortcut(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    index = int(param['from'])
    activation = param['activation']
    assert activation == 'linear'

    from_layer = stack[index]
    net = tf.add(net, from_layer, name=scope)
    return net


def cfg_yolo(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    output_index.append(((len(stack) - 1),param))
    return net


def cfg_upsample(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    stride = int(param['stride'])
    assert stride == 2
    net = upsample(net)
    return net


def cfg_upsample_org(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    stride = int(param['stride'])
    assert stride == 2

    net = tf.image.resize_nearest_neighbor(net, (H * stride, W * stride), name=scope)
    return net


def cfg_softmax(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    net = tf.squeeze(net, axis=[1, 2], name=scope+'/Squeeze')
    net = tf.nn.softmax(net, name=scope+'/Softmax')
    return net


def cfg_ignore(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    if verbose:
        print("=> Ignore: ", param)

    return net


_cfg_layer_dict = {
    "net": cfg_net,
    "convolutional": cfg_convolutional,
    "maxpool": cfg_maxpool,
    # "avgpool": cfg_avgpool,
    "route": cfg_route,
    "reorg": cfg_reorg,
    "shortcut": cfg_shortcut,
    "yolo": cfg_yolo,
    "upsample": cfg_upsample,
    "dropout": cfg_dropout,
    "softmax": cfg_softmax
}

# net 是当前的网络，可以获得当前的通道数
# output_index 是yolo层所在的位置
def get_cfg_layer(net, layer_name, param, weights_walker, stack, output_index,
                  scope=None, training=False, const_inits=True, verbose=True):
    B, H, W, C = [None, None, None, None] if net is None else net.shape.as_list()
    layer = _cfg_layer_dict.get(layer_name, cfg_ignore)(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose)
    return layer
