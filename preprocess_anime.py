import tensorflow as tf
import numpy as np
import os
from PIL import Image
import time
import pathlib
import string


# classes = ['ahoge','black_hair']

filenames = []

# raw_im_dir = "../brine_datasets/jayleicn/anime-faces/images"
raw_im_dir = "../anime_data/"


# for one_class in classes:
#     for image in os.listdir(raw_im_dir+'/{}'.format(one_class)):
#         if image.endswith(".jpg"):
#             filenames.append(raw_im_dir+'/{}/{}'.format(one_class,image))


for image in os.listdir(raw_im_dir):
    if image.endswith(".png"):
        filenames.append(raw_im_dir+'/{}'.format(image))

## If you want to reduce dataset size:
dataset_size_wanted = -1
filenames = filenames[:dataset_size_wanted]

np.random.shuffle(filenames)
n = len(filenames)
len_train = n*9//10




filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()

def modify_image():

    key,value = reader.read(filename_queue)

    images = tf.image.decode_jpeg(value)

    resized = tf.image.resize_image_with_crop_or_pad(images, 96,65)
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
    for dir_name in ["TrainB","TestB"]:
        pathlib.Path(obj_dir+"/"+dir_name).mkdir(parents=True, exist_ok=True)

    while i<len_train :
        f = open(obj_dir+"/TrainB/trainB{}.png".format(i), "wb+")
        f.write(res_image.eval())
        i+=1
        f.close()

    while i<n :
        f = open(obj_dir+"/TestB/testB{}.png".format(i), "wb+")
        f.write(res_image.eval())
        i+=1
        f.close()

    coord.request_stop()
    coord.join(threads)
