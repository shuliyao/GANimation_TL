import tensorflow as tf
import tensorlayer as tl


def generator(img_input, au_input, reuse=False):
    with tf.compat.v1.variable_scope('generator') as vs:
        if reuse:
            vs.reuse_variables()

        #yo has been represented as N arrays of size H*W(128*128)
        au = tf.expand_dims(au_input, axis=1, name='expand_dims1')
        au = tf.expand_dims(au, axis=2, name='expand_dims2')
        au = tf.tile(au, multiples=[1,128,128,1], name='tile')
        x = tf.concat([img_input, au], axis=3, name='concat')

        net = tl.layers.Input(x.shape(), tf.float32, name='input')
        w_init = tl.initializers.TruncatedNormal(0)
        b_init = tl.initializers.Constant(0.02)
        net = tl.layers.Conv2d(n_filter=64, filter_size=(7, 7), strides=(1, 1), W_init=w_init, b_init=b_init, name='conv1')(net)
        net = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_1')(net)

        net = tl.layers.Conv2d(n_filter=128, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=b_init, name='conv2')(
            net)
        net = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_2')(net)

        net = tl.layers.Conv2d(n_filter=256, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=b_init, name='conv3')(
            net)
        net = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_3')(net)

        for i in range(1, 7):
            with tf.compat.v1.variable_scope('resblock'+str(i)):
                net_ = tl.layers.Conv2d(n_filter=256, filter_size=(3,3), strides=(1,1), name='conv1')(net)
                net_ = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_1')(net_)
                net_ = tl.layers.Conv2d(n_filter=256, filter_size=(3,3), strides=(1,1), name='conv1')(net_)
                net_ = tl.layers.InstanceNorm2d(name='instance_norm_2')(net_)

                net = tf.add(net, net_)

        net = tl.layers.DeConv2d(n_filter=128, filter_size=(4,4), strides=(2,2), name='deconv1')(net)
        net = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_4')(net)

        net = tl.layers.DeConv2d(n_filter=64, filter_size=(4, 4), strides=(2, 2), name='deconv2')(net)
        net = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_5')(net)

        net = tl.layers.Conv2d(n_filter=3, filter_size=(7, 7), strides=(1, 1), name='conv4')(net)
        img_out = tf.tanh(net, name='tanh')

        net = tl.layers.Conv2d(n_filter=1, filter_size=(7, 7), strides=(1, 1), name='conv5')(net)
        mask_out = tf.sigmoid(net, name='sigmoid')

        return img_out, mask_out

def get_generator(shape):
    w_init = tl.initializers.TruncatedNormal(0)
    b_init = tl.initializers.Constant(0.02)
    ni = tl.layers.Input(shape=shape, name='input')
    net = tl.layers.Conv2d(n_filter=64, filter_size=(7, 7), strides=(1, 1), W_init=w_init, b_init=b_init, name='conv1')(ni)
    net = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_1')(net)
    net = tl.layers.Conv2d(n_filter=128, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=b_init, name='conv2')(net)
    net = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_2')(net)
    net = tl.layers.Conv2d(n_filter=256, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=b_init, name='conv3')(net)
    net = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_3')(net)
    for i in range(1, 7):
        net_old = net
        net = tl.layers.Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), name='conv1')(net)
        net = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_1')(net)
        net = tl.layers.Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), name='conv1')(net)
        net = tl.layers.InstanceNorm2d(name='instance_norm_2')(net)
        net = tf.add(net_old, net)
    net = tl.layers.DeConv2d(n_filter=128, filter_size=(4, 4), strides=(2, 2), name='deconv1')(net)
    net = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_4')(net)

    net = tl.layers.DeConv2d(n_filter=64, filter_size=(4, 4), strides=(2, 2), name='deconv2')(net)
    net = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_5')(net)

    net = tl.layers.Conv2d(n_filter=3, filter_size=(7, 7), strides=(1, 1), name='conv4')(net)
    img_out = tf.tanh(net, name='tanh')

    net = tl.layers.Conv2d(n_filter=1, filter_size=(7, 7), strides=(1, 1), name='conv5')(net)
    mask_out = tf.sigmoid(net, name='sigmoid')

    return tl.models.Model(inputs=ni, outputs=(img_out, mask_out))


def discriminator(input, reuse=False):
    with tf.compat.v1.variable_scope('discriminator') as vs:
        if reuse:
            vs.reuse_variables()

        net = tl.layers.Input(input.shape(), tf.float32, name='input')
        for i in range(1, 7):
            net = tl.layers.Conv2d(n_filter=64*(2**i), filter_size=(4,4), strides=(2,2), name='conv'+str(i+1))(net)
            net = tf.nn.leaky_relu(alpha=0.01, name='LReLU'+str(i+1))(net)

        img_out = tl.layers.Conv2d(n_filter=1, filter_size=(3, 3), strides=(1,1), name='conv7')
        au_out = tl.layers.Conv2d(n_filter=17, filter_size=(2, 2), strides=(1, 1), name='conv8')
        au_out = tf.squeeze(au_out, [1,2], name='squeeze1')
        return img_out, au_out

def get_discriminator(input_shape):

    ni = tl.layers.Input(input_shape, name='input')
    net = tl.layers.Conv2d(n_filter=64 * (2 ** 1), filter_size=(4, 4), strides=(2, 2),
                           name='conv1')(ni)
    net = tf.nn.leaky_relu(alpha=0.01, name='LReLU1')(net)

    for i in range(2, 7):
        net = tl.layers.Conv2d(n_filter=64 * (2 ** i), filter_size=(4, 4), strides=(2, 2),
                               name='conv' + str(i + 1))(net)
        net = tf.nn.leaky_relu(alpha=0.01, name='LReLU' + str(i + 1))(net)

    img_out = tl.layers.Conv2d(n_filter=1, filter_size=(3, 3), strides=(1, 1), name='conv7')(net)
    au_out = tl.layers.Conv2d(n_filter=17, filter_size=(2, 2), strides=(1, 1), name='conv8')(net)
    au_out = tf.squeeze(au_out, [1, 2], name='squeeze1')
    return tl.models.Model(inputs=ni, outputs=(img_out, au_out))










