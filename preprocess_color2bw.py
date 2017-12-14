import tensorflow as tf
import numpy as np
import os
from PIL import Image
import time
import pathlib


classes = ['bike','person','cars','none']

filenames = []
for one_class in classes:
    for image in os.listdir('../TUGraz/PNGImages/{}'.format(one_class)):
        filenames.append('../TUGraz/PNGImages/{}/{}'.format(one_class,image))
np.random.shuffle(filenames)

n = len(filenames)
len_train = n*9//10

filenames = filenames[:10]


filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()

def modify_image(bw=False):
    key,value = reader.read(filename_queue)
    images = tf.image.decode_png(value)

    resized = tf.image.resize_images(images, [120,160], 1)
    resized.set_shape([120,160,3])

    if bw:
        resized = tf.image.rgb_to_grayscale(resized)

    resized_encoded = tf.image.encode_png(resized,name="save_me")

    return resized_encoded

res_image = modify_image(bw=False)
res_image_bw = modify_image(bw=True)


obj_dir = "input/color2bw"

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners()

    i = 0
    for dir_name in ["TrainA","TrainB","TestA","TestB"]:
        pathlib.Path(obj_dir+"/"+dir_name).mkdir(parents=True, exist_ok=True)

    while i<len_train :
        if i < len_train/2:
            f = open(obj_dir+"/TrainA/trainA{}.png".format(i), "wb+")
            f.write(res_image.eval())
        else:
            f = open(obj_dir+"/TrainB/trainB{}.png".format(i), "wb+")
            f.write(res_image_bw.eval())
        i+=1
        f.close()

    while i<n :
        if i < len_train+(n-len_train)/2:
            f = open(obj_dir+"/TestA/testA{}.png".format(i), "wb+")
            f.write(res_image.eval())
        else:
            f = open(obj_dir+"/TestB/testB{}.png".format(i), "wb+")
            f.write(res_image_bw.eval())
        i+=1
        f.close()

    coord.request_stop()
    coord.join(threads)
