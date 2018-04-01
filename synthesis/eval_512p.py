from __future__ import division
import os, helper, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def lrelu(x):
    return tf.maximum(0.2 * x, x)


def build_net(ntype, nin, nwb=None, name=None):
    if ntype == 'conv':
        return tf.nn.relu(
            tf.nn.conv2d(
                nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) +
            nwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool(
            nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


def build_vgg19(input, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net = {}
    vgg_rawnet = scipy.io.loadmat('VGG_Model/imagenet-vgg-verydeep-19.mat')
    vgg_layers = vgg_rawnet['layers'][0]
    net['input'] = input - np.array([123.6800, 116.7790, 103.9390]).reshape(
        (1, 1, 1, 3))
    net['conv1_1'] = build_net(
        'conv',
        net['input'],
        get_weight_bias(vgg_layers, 0),
        name='vgg_conv1_1')
    net['conv1_2'] = build_net(
        'conv',
        net['conv1_1'],
        get_weight_bias(vgg_layers, 2),
        name='vgg_conv1_2')
    net['pool1'] = build_net('pool', net['conv1_2'])
    net['conv2_1'] = build_net(
        'conv',
        net['pool1'],
        get_weight_bias(vgg_layers, 5),
        name='vgg_conv2_1')
    net['conv2_2'] = build_net(
        'conv',
        net['conv2_1'],
        get_weight_bias(vgg_layers, 7),
        name='vgg_conv2_2')
    net['pool2'] = build_net('pool', net['conv2_2'])
    net['conv3_1'] = build_net(
        'conv',
        net['pool2'],
        get_weight_bias(vgg_layers, 10),
        name='vgg_conv3_1')
    net['conv3_2'] = build_net(
        'conv',
        net['conv3_1'],
        get_weight_bias(vgg_layers, 12),
        name='vgg_conv3_2')
    net['conv3_3'] = build_net(
        'conv',
        net['conv3_2'],
        get_weight_bias(vgg_layers, 14),
        name='vgg_conv3_3')
    net['conv3_4'] = build_net(
        'conv',
        net['conv3_3'],
        get_weight_bias(vgg_layers, 16),
        name='vgg_conv3_4')
    net['pool3'] = build_net('pool', net['conv3_4'])
    net['conv4_1'] = build_net(
        'conv',
        net['pool3'],
        get_weight_bias(vgg_layers, 19),
        name='vgg_conv4_1')
    net['conv4_2'] = build_net(
        'conv',
        net['conv4_1'],
        get_weight_bias(vgg_layers, 21),
        name='vgg_conv4_2')
    net['conv4_3'] = build_net(
        'conv',
        net['conv4_2'],
        get_weight_bias(vgg_layers, 23),
        name='vgg_conv4_3')
    net['conv4_4'] = build_net(
        'conv',
        net['conv4_3'],
        get_weight_bias(vgg_layers, 25),
        name='vgg_conv4_4')
    net['pool4'] = build_net('pool', net['conv4_4'])
    net['conv5_1'] = build_net(
        'conv',
        net['pool4'],
        get_weight_bias(vgg_layers, 28),
        name='vgg_conv5_1')
    net['conv5_2'] = build_net(
        'conv',
        net['conv5_1'],
        get_weight_bias(vgg_layers, 30),
        name='vgg_conv5_2')
    return net


def recursive_generator(label, sp):
    dim = 512 if sp >= 128 else 1024
    if sp == 512:
        dim = 128
    if sp == 4:
        input = label
    else:
        downsampled = tf.image.resize_area(
            label, (sp // 2, sp), align_corners=False)
        input = tf.concat([
            tf.image.resize_bilinear(
                recursive_generator(downsampled, sp // 2), (sp, sp * 2),
                align_corners=True), label
        ], 3)
    net = slim.conv2d(
        input,
        dim, [3, 3],
        rate=1,
        normalizer_fn=slim.layer_norm,
        activation_fn=lrelu,
        scope='g_' + str(sp) + '_conv1')
    net = slim.conv2d(
        net,
        dim, [3, 3],
        rate=1,
        normalizer_fn=slim.layer_norm,
        activation_fn=lrelu,
        scope='g_' + str(sp) + '_conv2')
    if sp == 512:
        net = slim.conv2d(
            net,
            3, [1, 1],
            rate=1,
            activation_fn=None,
            scope='g_' + str(sp) + '_conv100')
        net = (net + 1.0) / 2.0 * 255.0
    return net


def compute_error(real, fake, label):
    #return tf.reduce_sum(tf.reduce_mean(label*tf.expand_dims(tf.reduce_mean(tf.abs(fake-real),reduction_indices=[3]),-1),reduction_indices=[1,2]))#diversity loss
    return tf.reduce_mean(tf.abs(fake - real))  #simple loss


def run(images, is_training=False):
    sess = tf.Session()
    sp = 512  #spatial resolution: 512x1024
    with tf.variable_scope(tf.get_variable_scope()):
        label = tf.placeholder(tf.float32, [None, None, None, 20])
        real_image = tf.placeholder(tf.float32, [None, None, None, 3])
        fake_image = tf.placeholder(tf.float32, [None, None, None, 3])
        generator = recursive_generator(label, sp)
        weight = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("result_512p")
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver = tf.train.Saver(var_list=[
            var for var in tf.trainable_variables() if var.name.startswith('g_')
        ])
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        ckpt_prev = tf.train.get_checkpoint_state("result_256p")
        saver = tf.train.Saver(var_list=[
            var for var in tf.trainable_variables()
            if var.name.startswith('g_') and not var.name.startswith('g_512')
        ])
        print('loaded ' + ckpt_prev.model_checkpoint_path)
        saver.restore(sess, ckpt_prev.model_checkpoint_path)
    saver = tf.train.Saver(max_to_keep=1000)

    if not os.path.isdir("result_512p/final"):
        os.makedirs("result_512p/final")
    for img in images:
        if not os.path.isfile(img):
            continue
        semantic = helper.get_semantic_map(img)
        output = sess.run(
            generator,
            feed_dict={
                label:
                np.concatenate(
                    (semantic, np.expand_dims(
                        1 - np.sum(semantic, axis=3), axis=3)),
                    axis=3)
            })
        output = np.minimum(np.maximum(output, 0.0), 255.0)
        base = os.path.basename(img)
        fielname = os.path.splitext(base)[0]
        scipy.misc.toimage(
            output[0, :, :, :], cmin=0,
            cmax=255).save("result_512p/final/%s_output.jpg" % filename)
