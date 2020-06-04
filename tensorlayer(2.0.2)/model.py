import tensorflow as tf
import tensorlayer as tl

'''
class generator(tl.models.Model):

    def __init__(self):
        super(generator, self).__init__()

        w_init = tl.initializers.TruncatedNormal(0, 0.02)
        b_init = None
        gamma_init = tf.random_normal_initializer(1.0, 0.02)
        beta_init = tf.constant_initializer(0.0)

        self.conv1 = tl.layers.Conv2d(in_channels=20, n_filter=64, filter_size=(7, 7), strides=(1, 1), W_init=w_init, b_init=b_init, name='conv1')
        self.instance_norm1 = tl.layers.InstanceNorm2d(name='instance_norm_1', act=tf.nn.relu, gamma_init=gamma_init, beta_init=beta_init, num_features=64)

        self.conv2 = tl.layers.Conv2d(in_channels=64, n_filter=128, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=b_init, name='conv2')
        self.instance_norm2 = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_2', gamma_init=gamma_init, beta_init=beta_init, num_features=128)

        self.conv3 = tl.layers.Conv2d(in_channels=128, n_filter=256, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=b_init, name='conv3')
        self.instance_norm3 = tl.layers.InstanceNorm2d(act=tf.nn.relu, name='instance_norm_3', gamma_init=gamma_init, beta_init=beta_init, num_features=256)

    def forward(self, x):
        z = self.conv1(x)
        z = self.instance_norm1(z)
        z = self.conv2(z)
        z = self.instance_norm2(z)
        z = self.conv3(z)
        out = self.instance_norm3(z)
        return out
'''

def get_generator(shape):
    w_init = tl.initializers.TruncatedNormal(0, 0.02)
    b_init = None

    ni = tl.layers.Input(shape=shape, name='input')
    gamma_init = tf.random_normal_initializer(1.0, 0.02)
    bate_init = tf.constant_initializer(0.0)

    #tl.layers.set_name_reuse(reuse)
    #Down-Sampling
    net = tl.layers.Conv2d(n_filter=64, filter_size=(7, 7), strides=(1, 1), W_init=w_init, b_init=b_init, padding='SAME', name='conv1')(ni)
    net = tl.layers.BatchNorm2d(act=tf.nn.relu, name='instance_norm_1')(net)
    net = tl.layers.Conv2d(n_filter=128, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=b_init, name='conv2')(net)
    net = tl.layers.BatchNorm2d(act=tf.nn.relu, name='instance_norm_2')(net)
    net = tl.layers.Conv2d(n_filter=256, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=b_init, name='conv3')(net)
    net = tl.layers.BatchNorm2d(act=tf.nn.relu, name='instance_norm_3')(net)
    #resblock
    for i in range(1, 7):
        nn = tl.layers.Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), W_init=w_init, name='conv1_res'+str(i))(net)
        nn = tl.layers.BatchNorm2d(act=tf.nn.relu, name='instance_norm_1_res'+str(i))(nn)
        nn = tl.layers.Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), W_init=w_init, name='conv2_res'+str(i))(nn)
        nn = tl.layers.BatchNorm2d(name='instance_norm_2_res'+str(i))(nn)
        nn = tl.layers.Elementwise(combine_fn=tf.add, name='elementwise1_res'+str(i))([net, nn])
        net = nn
    #Up-Sampling
    net = tl.layers.DeConv2d(n_filter=128, filter_size=(4, 4), strides=(2, 2), W_init=w_init, name='deconv1')(net)
    net = tl.layers.BatchNorm2d(act=tf.nn.relu, name='instance_norm_4')(net)
    net = tl.layers.DeConv2d(n_filter=64, filter_size=(4, 4), strides=(2, 2), W_init=w_init, name='deconv2')(net)
    net = tl.layers.BatchNorm2d(act=tf.nn.relu, name='instance_norm_5')(net)

    img_out = tl.layers.Conv2d(n_filter=3, filter_size=(7, 7), act=tf.tanh, strides=(1, 1), W_init=w_init, name='conv4')(net)
    mask_out = tl.layers.Conv2d(n_filter=1, filter_size=(7, 7), act=tf.sigmoid, strides=(1, 1), W_init=w_init, name='conv5')(net)

    return tl.models.Model(inputs=(ni), outputs=([img_out, mask_out]))



def get_discriminator(input_shape):
    w_init = tl.initializers.TruncatedNormal(0, 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.01)

    #tl.layers.set_name_reuse(reuse)
    ni = tl.layers.Input(input_shape, name='input')
    net = ni
    for i in range(6):
        net = tl.layers.Conv2d(n_filter=64*(2**i), filter_size=(4, 4), strides=(2, 2), W_init=w_init, act=lrelu, name='conv' + str(i+1))(net)

    img_out = tl.layers.Conv2d(n_filter=1, filter_size=(3, 3), strides=(1, 1), W_init=w_init, name='conv7')(net)
    au_out = tl.layers.Conv2d(n_filter=17, filter_size=(2, 2), strides=(1, 1), W_init=w_init, padding='VALID', name='conv8')(net)
    #au_out = tf.squeeze(input=au_out, axis=[1, 2], name='squeeze1')
    return tl.models.Model(inputs=(ni), outputs=([img_out, au_out]))










