import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import cv2
import face_recognition
from model import get_generator, get_discriminator
from tqdm import tqdm

def preprocess_data(img_input, au_input):
    au = tf.expand_dims(au_input, axis=1, name='expand_dims1')#[None, 1, 17]
    au = tf.expand_dims(au, axis=2, name='expand_dims2')#[None, 1, 1, 17]
    au = tf.tile(au, multiples=[1, 128, 128, 1], name='tile')#[None, 128, 128, 17]
    x = tf.concat([img_input, au], axis=3, name='concat')#[None, 128, 128, 20]
    return x

if __name__ == '__main__':
    G = get_generator([None, 128, 128, 20])
    D = get_discriminator([None, 128, 128, 3])

    G_path = 'G_30.npz'
    tl.files.load_and_assign_npz(name=G_path, network=G)
    D_path = 'D_30.npz'
    tl.files.load_and_assign_npz(name=D_path, network=D)

    imgs_names = os.listdir('test_face')
    real_src = face_recognition.load_image_file('obama.jpeg')  # RGB image
    face_loc = face_recognition.face_locations(real_src)

    top, right, bottom, left = face_loc[0]
    if len(face_loc) == 1:
        top, right, bottom, left = face_loc[0]

    real_face = np.zeros((1, 128, 128, 3), dtype=np.float32)
    style_face = np.zeros((1, 128, 128, 3), dtype=np.float32)
    real_face[0] = cv2.resize(real_src[top:bottom, left:right], (128, 128)) / 127.5 - 1

    for img_name in tqdm(imgs_names):#将原始图像变为style image的表情
        style_face[0] = cv2.imread('test_face/'+img_name)[:,:,::-1] / 127.5 - 1
        #print(sess.run(desired_au, feed_dict={style_img:style_face}))
        #output = sess.run(fake_img_masked, feed_dict={real_img:real_face, style_img:style_face})

        _, desired_au = D(style_face)
        x = preprocess_data(real_face, desired_au)
        [fake_img, fake_mask] = G(x)
        output = fake_mask * real_face + (1 - fake_mask) * fake_img

        real_src[top:bottom, left:right] = cv2.resize((output[0]+1)*127.5, (right-left,bottom-top))
        cv2.imwrite('output/'+img_name, real_src[:,:,::-1])
