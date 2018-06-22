import logging

import tensorflow as tf

import resnet_v1
import resnet_utils

from resnet_v1 import bottleneck

log = logging.getLogger()
slim = tf.contrib.slim

RESNET_MEAN = [103.062623801, 115.902882574, 123.151630838, ]
CKPT = './resnet/resnet50_full.ckpt'
DEFAULT_SCOPE = 'resnet_v1_50'


def create_trunk(images, rois=None, reuse=False,
                 fc_layers=True, weight_decay=0.0005):
    red, green, blue = tf.split(images*255, 3, axis=3)
    images = tf.concat([blue, green, red], 3) - RESNET_MEAN

    with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=False,
                                                   weight_decay=weight_decay)):
        net, endpoints = resnet_frcnn(images, rois=rois, global_pool=True,
                                      fc_layers=fc_layers, reuse=reuse)
    return net, endpoints


def resnet_frcnn(inputs,
                 rois=None,
                 global_pool=True,
                 reuse=None,
                 fc_layers=True,
                 scope='resnet_v1_50'):
    blocks = [
      resnet_utils.Block(
          'block1', bottleneck, [(256, 64, 1)] * 3),
      resnet_utils.Block(
          'block2', bottleneck, [(512, 128, 2)] + [(512, 128, 1)] * 3),
      resnet_utils.Block(
          'block3', bottleneck, [(1024, 256, 2)] + [(1024, 256, 1)] * 5),
      resnet_utils.Block(
          'block4', bottleneck, [(2048, 512, 1)] + [(2048, 512, 1)] * 2)
    ]

    if rois is None:
        log.warning("No RoI transmitted, recreating normal ResNet")
        if not fc_layers:
            blocks = blocks[:-1]
            global_pool = False
        else:
            blocks = blocks[:-1] + [resnet_utils.Block(
                'block4', bottleneck, [(2048, 512, 2)] + [(2048, 512, 1)] * 2)]
        net, endpoints = resnet_v1.resnet_v1(inputs, blocks,
                                             global_pool=global_pool,
                                             reuse=reuse, scope=scope)
    else:
        if not fc_layers:
            raise NotImplementedError
        net = inputs
        net, ep1 = resnet_v1.resnet_v1(net, blocks[:-1], global_pool=False,
                                       reuse=reuse, scope=scope)

        z = tf.zeros(tf.stack([tf.shape(rois)[0]]), dtype=tf.int32)
        net = tf.image.crop_and_resize(net, rois, z, [7, 7], name="roi_warping")

        net, ep2 = resnet_v1.resnet_v1(net, blocks[-1:],
                                       global_pool=global_pool,
                                       include_root_block=False,
                                       reuse=reuse, scope=scope)
        if global_pool:
            net = slim.flatten(net)
        endpoints = ep1.copy()
        endpoints.update(ep2)
        # endpoints = {**ep1, **ep2}  # python3.5, fix it when we ditch fedora

    return net, endpoints


def get_imagenet_init():
    variables = slim.get_model_variables(scope=DEFAULT_SCOPE)
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(CKPT, variables)
    return init_assign_op, init_feed_dict
