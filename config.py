# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pylab as pylab
import seaborn as sns
from google.colab import drive
import itertools
import shutil
import cv2
import os
import sys
import json
import csv
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
np.set_printoptions(precision=2)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 15),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Constant

if os.getcwd().startswith('/Users/tal'):
    path = '/Users/tal/Dropbox/Projects/Facial_Attributes_Detection'
    MODEL_PATH = '/Users/tal/Dropbox/Projects/'
    WEIGHT_PATH = '/Users/tal/Dropbox/Projects/Facial_Attributes_Detection/Other/Weights'
    IMAGE_PATH = '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/1'
    IND_FILE = '/Users/tal/Google Drive/Cellebrite/files list.csv'
    PATH_JSON = '/Users/tal/Dropbox/Projects/Facial_Attributes_Detection/json'
    PATH_CSV = '/Users/tal/Dropbox/Projects/Facial_Attributes_Detection/csv/data'
    EMB_PATH = '/Users/tal/Dropbox/Projects/Facial_Attributes_Detection/csv/emb'

elif os.getcwd().startswith('/Users/Sheryl'):
    IMAGE_PATH = '/Users/Sheryl/Desktop/ITC2/Cellebrite Project/face_att_sheryl'
    IND_FILE = '/Users/Sheryl/Desktop/ITC2/Cellebrite Project/files_list_sheryl.csv'
    WEIGHT_PATH = '/Users/Sheryl/PycharmProjects/Facial_Attributes_Detection_BEFORE/Weights'
else:
    IMAGE_PATH = '/home/ubuntu/sheryl/S3/facial-attributes-itc/face_att_sheryl'
    IND_FILE = '/home/ubuntu/sheryl/S3/facial-attributes-itc/files_list_sheryl.csv'
    WEIGHT_PATH = '/home/ubuntu/Facial_Attributes_Detection/weights'
    MODEL_PATH = '/home/ubuntu/Facial_Attributes_Detection/'
    EMB_PATH = '/home/ubuntu/Facial_Attributes_Detection/csv/emb'


def find_imagepath(file):
    """
    function look for image files in the relevant directory determine by condition
    param: file - str name of image file
    return: relevent file path
    """
    temp_file = int(file.split('.')[0][-6:])
    if temp_file <= 70000:
        IMAGEPATH = '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/1/'
    elif 70000 < temp_file < 140000:
        IMAGEPATH = '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/2/'
    elif temp_file >= 140000:
        IMAGEPATH = '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/3/'
    else:
        IMAGEPATH = '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/'
    return os.path.join(IMAGEPATH, file)


# Constant For CelebA DS
FILE = ''
if os.getcwd() != '/content':
    IMAGEPATH = 'Datasets'
    IND_FILE_DS = '/Users/tal/Google Drive/Cellebrite/files list.csv'
    try:
        FACEPATH = os.path.join(IMAGEPATH, 'face_att')
    except:
        FACEPATH = find_imagepath(FILE)
else:
    drive.mount('/content/drive')
    IND_FILE_DS = '/content/drive/My Drive/Cellebrite/files list.csv'
    IMAGEPATH = '/content/drive/My Drive/Cellebrite/Datasets'
    FACEPATH = os.path.join(IMAGEPATH, 'face_att')

# Constant For Age-Race-Gender DS
AGE = -4
SEX = -3
RACE = -2
IMAGEPATH2 = '/Users/tal/Downloads/Databases/age_sex_race/'

# Constant For Facial Expression
FILEPATH3 = '/Users/tal/Downloads/Databases/facial_expressions-master/legend.csv'
IMAGEPATH3 = '/Users/tal/Downloads/Databases/facial_expressions-master/images'

EPOCH = 1
