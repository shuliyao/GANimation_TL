import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from model import get_generator, get_discriminator
from data import load_data
import time
import pickle


# hyper-parameters
face_dir = 'data/imgs_celebA/'
au_dir = 'data/aus_openface.pkl'

BATCH_SIZE = 25
EPOCHS = 30

lambda_D_img = 1
lambda_D_au = 4000
lambda_D_gp = 10
lambda_cyc = 10
lambda_mask = 0.1
lambda_mask_smooth = 1e-5


def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def l2_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def smooth_loss(mat):
    return tf.reduce_sum(tf.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :])) + \
           tf.reduce_sum(tf.abs(mat[:, :-1, :, :] - mat[:, 1:, :, :]))


def preprocess_data(img_input, au_input):
    au = tf.expand_dims(au_input, axis=1, name='expand_dims1')#[None, 1, 17]
    au = tf.expand_dims(au, axis=2, name='expand_dims2')#[None, 1, 1, 17]
    au = tf.tile(au, multiples=[1, 128, 128, 1], name='tile')#[None, 128, 128, 17]
    x = tf.concat([img_input, au], axis=3, name='concat')#[None, 128, 128, 20]
    return x


def get_data(img_input, au_input):
    length = len(img_input)
    def preprocess_data(img_input, au_input):
        au = tf.expand_dims(au_input, axis=1, name='expand_dims1')  # [None, 1, 17]
        au = tf.expand_dims(au, axis=2, name='expand_dims2')  # [None, 1, 1, 17]
        au = tf.tile(au, multiples=[1, 128, 128, 1], name='tile')  # [None, 128, 128, 17]
        x = tf.concat([img_input, au], axis=3, name='concat')  # [None, 128, 128, 20]
        return x

    train_ds = tf.data.Dataset.from_generator(preprocess_data(img_input, au_input), output_types=tf.float32)
    ds = train_ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=2)
    return ds, length

#train
def train():
    #construct data
    real_img = tf.compat.v1.placeholder(tf.float32, [None, 128, 128, 3], name='real_img')
    real_au = tf.compat.v1.placeholder(tf.float32, [None, 17], name='real_au')
    desired_au = tf.compat.v1.placeholder(tf.float32, [None, 17], name='desired_au')
    lr = tf.compat.v1.placeholder(tf.float32, name='lr')

    #load data
    face, au = load_data(face_dir, au_dir)
    au_rand = au.copy()
    np.random.shuffle(au_rand)
    au_rand += np.random.uniform(-0.1, 0.1, au_rand.shape)

    G = get_generator([None, 128, 128, 20])
    D = get_discriminator([None, 128, 128, 3])

    G.train()
    D.train()

    g_train_op = tf.optimizers.Adam(learning_rate=lr, beta1=0.5, beta2=0.999)
    d_train_op = tf.optimizers.Adam(learning_rate=lr, beta1=0.5, beta2=0.999)

    d_loss = 0
    g_loss = 0
    counter = 1

    n_step_epoch = int(len(face) // BATCH_SIZE)
    num_tiles = int(np.sqrt(25))
    print('----------- start training -----------')
    for e in range(1, EPOCHS+1):
        start_time = time.time()
        #lr设置
        if e <= 21:
            lr_now = 1e-4
        else:
            lr_now = 1e-5 * (EPOCHS + 1 - e)
        print('===== [Epoch %02d/30](lr: %.5f) =====' % (e, lr_now))

        for i in range(len(face) // BATCH_SIZE):
            with tf.GradientTape(persistent=True) as tape:
                real_img = face[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                print(real_img.shape())
                real_au = au[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                print(real_au.shape())
                desired_au = au_rand[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                print(desired_au.shape())
                lr = lr_now

                g_input_f = preprocess_data(real_img, desired_au)
                fake_img, fake_mask = G(g_input_f)
                fake_img_masked = fake_mask * real_img + (1 - fake_mask) * fake_img

                g_input_cyc = preprocess_data(fake_img_masked, real_au)
                cyc_img, cyc_mask = G(g_input_cyc)
                cyc_img_masked = cyc_mask * fake_img_masked + (1 - cyc_mask) * cyc_img

                # D(real_I)
                pred_real_img, pred_real_au = D(real_img, reuse=False)
                # D(fake_I)
                pred_fake_img_masked, pred_fake_au = D(fake_img_masked, reuse=True)

                # loss
                loss_g_fake_img_masked = -tf.reduce_mean(pred_fake_img_masked) * lambda_D_img
                loss_g_fake_au = l2_loss(desired_au, pred_fake_au) * lambda_D_au
                loss_g_cyc = l1_loss(real_img, cyc_img_masked) * lambda_cyc

                loss_g_mask_fake = tf.reduce_mean(fake_mask) * lambda_mask + smooth_loss(fake_mask) * lambda_mask_smooth
                loss_g_mask_cyc = tf.reduce_mean(cyc_mask) * lambda_mask + smooth_loss(cyc_mask) * lambda_mask_smooth

                loss_g = loss_g_fake_img_masked + loss_g_fake_au + \
                         loss_g_cyc + \
                         loss_g_mask_fake + loss_g_mask_cyc

                loss_d_img = -tf.reduce_mean(pred_real_img) * lambda_D_img + tf.reduce_mean(
                    pred_fake_img_masked) * lambda_D_img
                loss_d_au = l2_loss(real_au, pred_real_au) * lambda_D_au

                alpha = tf.compat.v1.random_uniform([BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.)
                differences = fake_img_masked - real_img
                interpolates = real_img + tf.multiply(alpha, differences)
                gradients = tf.gradients(D(interpolates, reuse=True), [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                loss_d_gp = lambda_D_gp * gradient_penalty

                loss_d = loss_d_img + loss_d_au + loss_d_gp

                real_img_sum = tf.summary.image('real_img', real_img)

                fake_img_sum = tf.summary.image('fake_img', fake_img)
                fake_mask_sum = tf.summary.image('fake_mask', fake_mask)
                fake_img_masked_sum = tf.summary.image('fake_img_masked', fake_img_masked)

                cyc_img_sum = tf.summary.image('cyc_img', cyc_img)
                cyc_mask_sum = tf.summary.image('cyc_mask', cyc_mask)
                cyc_img_masked_sum = tf.summary.image('cyc_img_masked', cyc_img_masked)

                loss_g_sum = tf.summary.scalar('loss_g', loss_g)
                loss_g_fake_img_masked_sum = tf.summary.scalar('loss_g_fake_img_masked', loss_g_fake_img_masked)
                loss_g_fake_au_sum = tf.summary.scalar('loss_g_fake_au', loss_g_fake_au)
                loss_g_cyc_sum = tf.summary.scalar('loss_g_cyc', loss_g_cyc)
                loss_g_mask_fake_sum = tf.summary.scalar('loss_g_mask_fake', loss_g_mask_fake)
                loss_g_mask_cyc_sum = tf.summary.scalar('loss_g_mask_cyc', loss_g_mask_cyc)

                loss_d_sum = tf.summary.scalar('loss_d', loss_d)
                loss_d_img_sum = tf.summary.scalar('loss_d_img', loss_d_img)
                loss_d_au_sum = tf.summary.scalar('loss_d_au', loss_d_au)
                loss_d_gp_sum = tf.summary.scalar('loss_d_gp', loss_d_gp)

            grad = tape.gradient(d_loss, D.trainable_weights)
            d_train_op.apply_gradients(zip(grad, D.trainable_weights))
            del tape

            d_summary_str = "Epoch: [{}/{}] [{}/{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}".format(e,
                                                                                         EPOCHS, i,
                                                                                         n_step_epoch,
                                                                                         time.time() - start_time,
                                                                                         d_loss, g_loss)
            if (i+1) % 5 == 0:
                with tf.GradientTape(persistent=True) as tape:
                    real_img = face[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                    real_au = au[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                    desired_au = au_rand[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                    lr = lr_now

                    g_input_f = preprocess_data(real_img, desired_au)
                    fake_img, fake_mask = G(g_input_f, reuse=False)
                    fake_img_masked = fake_mask * real_img + (1 - fake_mask) * fake_img
                    # G(G(Ic1, c2)*M, c1) * M
                    g_input_cyc = preprocess_data(fake_img_masked, real_au)
                    cyc_img, cyc_mask = G(g_input_cyc, reuse=True)
                    cyc_img_masked = cyc_mask * fake_img_masked + (1 - cyc_mask) * cyc_img

                    # D(real_I)
                    pred_real_img, pred_real_au = D(real_img, reuse=False)
                    # D(fake_I)
                    pred_fake_img_masked, pred_fake_au = D(fake_img_masked, reuse=True)

                    # loss
                    loss_g_fake_img_masked = -tf.reduce_mean(pred_fake_img_masked) * lambda_D_img
                    loss_g_fake_au = l2_loss(desired_au, pred_fake_au) * lambda_D_au
                    loss_g_cyc = l1_loss(real_img, cyc_img_masked) * lambda_cyc

                    loss_g_mask_fake = tf.reduce_mean(fake_mask) * lambda_mask + smooth_loss(
                        fake_mask) * lambda_mask_smooth
                    loss_g_mask_cyc = tf.reduce_mean(cyc_mask) * lambda_mask + smooth_loss(
                        cyc_mask) * lambda_mask_smooth

                    loss_g = loss_g_fake_img_masked + loss_g_fake_au + \
                             loss_g_cyc + \
                             loss_g_mask_fake + loss_g_mask_cyc

                    loss_d_img = -tf.reduce_mean(pred_real_img) * lambda_D_img + tf.reduce_mean(
                        pred_fake_img_masked) * lambda_D_img
                    loss_d_au = l2_loss(real_au, pred_real_au) * lambda_D_au

                    alpha = tf.compat.v1.random_uniform([BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.)
                    differences = fake_img_masked - real_img
                    interpolates = real_img + tf.multiply(alpha, differences)
                    gradients = tf.gradients(D(interpolates, reuse=True), [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
                    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                    loss_d_gp = lambda_D_gp * gradient_penalty

                    loss_d = loss_d_img + loss_d_au + loss_d_gp

                    real_img_sum = tf.summary.image('real_img', real_img)

                    fake_img_sum = tf.summary.image('fake_img', fake_img)
                    fake_mask_sum = tf.summary.image('fake_mask', fake_mask)
                    fake_img_masked_sum = tf.summary.image('fake_img_masked', fake_img_masked)

                    cyc_img_sum = tf.summary.image('cyc_img', cyc_img)
                    cyc_mask_sum = tf.summary.image('cyc_mask', cyc_mask)
                    cyc_img_masked_sum = tf.summary.image('cyc_img_masked', cyc_img_masked)

                    loss_g_sum = tf.summary.scalar('loss_g', loss_g)
                    loss_g_fake_img_masked_sum = tf.summary.scalar('loss_g_fake_img_masked', loss_g_fake_img_masked)
                    loss_g_fake_au_sum = tf.summary.scalar('loss_g_fake_au', loss_g_fake_au)
                    loss_g_cyc_sum = tf.summary.scalar('loss_g_cyc', loss_g_cyc)
                    loss_g_mask_fake_sum = tf.summary.scalar('loss_g_mask_fake', loss_g_mask_fake)
                    loss_g_mask_cyc_sum = tf.summary.scalar('loss_g_mask_cyc', loss_g_mask_cyc)

                    loss_d_sum = tf.summary.scalar('loss_d', loss_d)
                    loss_d_img_sum = tf.summary.scalar('loss_d_img', loss_d_img)
                    loss_d_au_sum = tf.summary.scalar('loss_d_au', loss_d_au)
                    loss_d_gp_sum = tf.summary.scalar('loss_d_gp', loss_d_gp)

                grad = tape.gradient(g_loss, G.trainable_weights)
                g_train_op.apply_gradients(zip(grad, G.trainable_weights))
                del tape

                g_summary_str = "Epoch: [{}/{}] [{}/{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}".format(e,
                                                                                                           EPOCHS, i,
                                                                                                           n_step_epoch,
                                                                                                           time.time() - start_time,
                                                                                                           d_loss,
                                                                                                           g_loss)
            counter += 1

        print('(spend time: %.2fmin) loss_g: %.4f  loss_d: %.4f \n' %
              ((time.time()-start_time)/60, g_loss, d_loss))

        if np.mod(e, 10) == 0:
            G.save_weights('{}/G.npz'.format('checkpoint'), format='npz')
            D.save_weights('{}/D.npz'.format('checkpoint'), format='npz')
            G.eval()
            result = G(g_input_f)
            G.train()
            tl.visualize.save_images(result.numpy(), [num_tiles, num_tiles],
                                     '{}/train_{:02d}.png'.format('samples', e))


if __name__ == '__main__':
    face, au = load_data(face_dir, au_dir)
    '''
    test
    aus_pkl = os.path.join('data/', 'aus_openface.pkl')
    print(aus_pkl)
    aus_dict = load_dict('D://Projects//GANimation-tl-master//data//aus_openface.pkl')
    assert os.path.isfile('data/imgs_celebA/000001.jpg'), "Cannot find image file: %s" % 'data/imgs_celebA'
    img_id = str(os.path.splitext(os.path.basename('data/imgs_celebA/000001.jpg'))[0])
    src_aus = aus_dict[img_id] / 5.0  # norm to [0, 1]
    print(src_aus)
    '''
    train()
