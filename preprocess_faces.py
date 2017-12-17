import tensorflow as tf
import numpy as np
import os
from PIL import Image
import time
import pathlib

filenames = []

raw_im_dir = "../cropped_faces"


for image in os.listdir(raw_im_dir+'/'):
    filenames.append(raw_im_dir+'/{}'.format(image))

np.random.shuffle(filenames)
## If you want to reduce dataset size:
dataset_size_wanted = -1
filenames = filenames[:dataset_size_wanted]

n = len(filenames)
len_train = n*9//10

filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()

def modify_image():
    key,value = reader.read(filename_queue)
    images = tf.image.decode_jpeg(value)

    resized = tf.image.resize_images(images, [96,65], 1)
    resized.set_shape([96,65,3])

    resized_encoded = tf.image.encode_png(resized,name="save_me")

    return resized_encoded

res_image = modify_image()



obj_dir = "input/face2anime"

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners()

    i = 0
    for dir_name in ["TrainA","TestA"]:
        pathlib.Path(obj_dir+"/"+dir_name).mkdir(parents=True, exist_ok=True)

    while i<len_train :
        f = open(obj_dir+"/TrainA/trainA{}.png".format(i), "wb+")
        f.write(res_image.eval())
        i+=1
        f.close()

    while i<n :
        f = open(obj_dir+"/TestA/testA{}.png".format(i), "wb+")
        f.write(res_image.eval())
        i+=1
        f.close()

    coord.request_stop()
    coord.join(threads)
