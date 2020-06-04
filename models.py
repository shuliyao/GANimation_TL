'''
Implement using tensorlayer 1.11.0
'''

import tensorflow as tf
import tensorlayer as tl


def generator(real_img, desired_au, reuse=False):
    w_init = tf.truncated_normal_initializer(0, 0.02)
    b_init = None
    gamma_init = tf.random_normal_initializer(1.0, 0.02)
    bate_init = tf.constant_initializer(0.0)

    with tf.variable_scope('Generator', reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        #au[None, 17] -> au[None, 128, 128, 17]
        desired_au = tf.expand_dims(desired_au, axis=1, name='ExpandDims1')
        desired_au = tf.expand_dims(desired_au, axis=2, name='ExpandDims2')
        desired_au = tf.tile(desired_au, multiples=[1, 128, 128, 1], name='Tile')
        x = tf.concat([real_img, desired_au], axis=3, name='Concat')#x[None, 128, 128, 20]

        #Down-Sampling
        ni = tl.layers.InputLayer(x, name='input')
        net = tl.layers.Conv2d(ni, n_filter=64, filter_size=(7, 7), strides=(1, 1), W_init=w_init, b_init=b_init, padding='SAME', name='conv1')
        net = tl.layers.InstanceNormLayer(net, act=tf.nn.relu, name='instance_norm_1')
        net = tl.layers.Conv2d(net, n_filter=128, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=b_init, name='conv2')
        net = tl.layers.InstanceNormLayer(net, act=tf.nn.relu, name='instance_norm_2')
        net = tl.layers.Conv2d(net, n_filter=256, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=b_init, name='conv3')
        net = tl.layers.InstanceNormLayer(net, act=tf.nn.relu, name='instance_norm_3')
        #resblock
        for i in range(1, 7):
            nn = tl.layers.Conv2d(net, n_filter=256, filter_size=(3, 3), strides=(1, 1), W_init=w_init, name='conv1_res'+str(i))
            nn = tl.layers.InstanceNormLayer(nn, act=tf.nn.relu, name='instance_norm_1_res'+str(i))
            nn = tl.layers.Conv2d(nn, n_filter=256, filter_size=(3, 3), strides=(1, 1), W_init=w_init, name='conv2_res'+str(i))
            nn = tl.layers.InstanceNormLayer(nn, name='instance_norm_2_res'+str(i))
            nn = tl.layers.ElementwiseLayer([net, nn],combine_fn=tf.add, name='elementwise1_res'+str(i))
            net = nn
        #Up-Sampling
        net = tl.layers.DeConv2d(net, n_out_channel=128, filter_size=(4, 4), strides=(2, 2), W_init=w_init, name='deconv1')
        net = tl.layers.InstanceNormLayer(net, act=tf.nn.relu, name='instance_norm_4')
        net = tl.layers.DeConv2d(net, n_out_channel=64, filter_size=(4, 4), strides=(2, 2), W_init=w_init, name='deconv2')
        net = tl.layers.InstanceNormLayer(net, act=tf.nn.relu, name='instance_norm_5')

        img_out = tl.layers.Conv2d(net, n_filter=3, filter_size=(7, 7), act=tf.tanh, strides=(1, 1), W_init=w_init, name='conv4')
        mask_out = tl.layers.Conv2d(net, n_filter=1, filter_size=(7, 7), act=tf.sigmoid, strides=(1, 1), W_init=w_init, name='conv5')

        return img_out.outputs, mask_out.outputs



def discriminator(x, reuse=False):
    w_init = tf.truncated_normal_initializer(0, 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.01)

    with tf.variable_scope('Discriminator', reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        ni = tl.layers.InputLayer(x, name='input')
        net = ni
        for i in range(6):
            net = tl.layers.Conv2d(net, n_filter=64*(2**i), filter_size=(4, 4), strides=(2, 2), W_init=w_init, act=lrelu, name='conv' + str(i+1))

        img_out = tl.layers.Conv2d(net, n_filter=1, filter_size=(3, 3), strides=(1, 1), W_init=w_init, name='conv7')
        au_out = tl.layers.Conv2d(net, n_filter=17, filter_size=(2, 2), strides=(1, 1), W_init=w_init, padding='VALID', name='conv8')
        au_out = tf.squeeze(input=au_out.outputs, axis=[1, 2], name='squeeze1')
        return img_out.outputs, au_out