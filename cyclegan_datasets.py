"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'horse2zebra_train': 1334,
    'horse2zebra_test': 140, 
    'color2bw_train' : 1328,
    'color2bw_test' : 148 ,
    'face2anime_train' : 1248,
    'color2bw_test' : 106 
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'horse2zebra_train': '.jpg',
    'horse2zebra_test': '.jpg',
    'color2bw_train': '.png',
    'color2bw_test': '.png',
    'face2anime_train': '.png',
    'face2anime_test': '.png'
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'horse2zebra_train': './cyclegan-1/input/horse2zebra/horse2zebra_train.csv',
    'horse2zebra_test': './cyclegan-1/input/horse2zebra/horse2zebra_test.csv',
    'color2bw_train': './cyclegan-1/input/color2bw/color2bw_train.csv',
    'color2bw_test': './cyclegan-1/input/color2bw/color2bw_test.csv',
    'face2anime_train': './cyclegan-1/input/face2anime/face2anime_train.csv',
    'face2anime_test': './cyclegan-1/input/face2anime/face2anime_test.csv'
}
