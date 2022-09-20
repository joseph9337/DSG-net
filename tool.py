import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def total_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("number of trainable parameters: %d"%(total_parameters))

def setup_summary(list):
    variables = []
    for i in range(len(list)):
        variables.append(tf.Variable(0.))
        tf.summary.scalar(list[i], variables[i])
    summary_vars = [x for x in variables]
    summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op



def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]

def load_image(path):
    img = Image.open(path)
    return img

def preprocess_image_RF(cv_img):
    img = np.array(cv_img, dtype=np.float32)
    img = img/255
    return img



def preprocess_image_16bit(cv_img):
    cv_img = cv_img.resize((128,128))
    img = np.array(cv_img, dtype=np.float32)
    img = img/65535
    return img

def preprocess_image(cv_img):
    img = np.array(cv_img, dtype=np.float32)
    img = img/255
    return img




def write_log(callback, names, logs, batch_no):
    """
    Util to write callback for Keras training
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()



def AUG_RF(data1,data2,data3,data4,data5,data6,data7,label, edge):
    data1 = np.reshape(data1,[256,3018])
    data2 = np.reshape(data2, [256, 3018])
    data3 = np.reshape(data3, [256, 3018])
    data4 = np.reshape(data4, [256, 3018])
    data5 = np.reshape(data5, [256, 3018])
    data6 = np.reshape(data6, [256, 3018])
    data7 = np.reshape(data7, [256, 3018])
    label = np.reshape(label, [128, 128])
    edge = np.reshape(edge, [128, 128])

    Aug_data7 =cv2.flip(data1,0)
    Aug_data6 = cv2.flip(data2, 0)
    Aug_data5 = cv2.flip(data3, 0)
    Aug_data4 = cv2.flip(data4, 0)
    Aug_data3 = cv2.flip(data5, 0)
    Aug_data2 = cv2.flip(data6, 0)
    Aug_data1 = cv2.flip(data7, 0)
    Aug_label = cv2.flip(label, 0)
    Aug_edge =  cv2.flip(edge, 0)

    Aug_data1=np.reshape(Aug_data1,[1,256,3018,1])
    Aug_data2 = np.reshape(Aug_data2, [1, 256, 3018, 1])
    Aug_data3 = np.reshape(Aug_data3, [1, 256, 3018, 1])
    Aug_data4 = np.reshape(Aug_data4, [1, 256, 3018, 1])
    Aug_data5 = np.reshape(Aug_data5, [1, 256, 3018, 1])
    Aug_data6 = np.reshape(Aug_data6, [1, 256, 3018, 1])
    Aug_data7 = np.reshape(Aug_data7, [1, 256, 3018, 1])
    Aug_label = np.reshape(Aug_label, [1, 128, 128, 1])
    Aug_edge = np.reshape(Aug_edge, [1, 128, 128, 1])

    return Aug_data1,Aug_data2,Aug_data3,Aug_data4,Aug_data5,Aug_data6,Aug_data7,Aug_label, Aug_edge


from math import log10, sqrt

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def VSG_AUG(cv_img, sub_ind, cp_ind):

    sz= np.shape(cv_img)[0]
    snz = sz/2
    cv_img = cv_img[round(snz-0.5-snz*sub_ind):round(snz-0.5+snz*sub_ind), :]
    cv_img = np.array(cv_img, dtype=np.float32)
    cv_img =cv2.resize(cv_img, dsize=(3018, round(np.shape(cv_img)[0]*cp_ind)),interpolation=cv2.INTER_NEAREST)
    cv_img =cv2.resize(cv_img, dsize=(3018, 256))
    img = np.array(cv_img, dtype=np.float32)
    img = img/255
    img=np.reshape(img,(256,3018,1))
    return img

import random
def Data_load(RF_v1, RF_v2, RF_v3,RF_v4, BM_v1,BM_v2,BM_v3,BM_v4,label_mt):
    ind_mt1 = random.randrange(0, 4)
    ind_mt2 = random.randrange(0, 4)
    ind_mt3 = random.randrange(0, 4)

    if ind_mt1 == 0:
        RF_SRCA = RF_v1
        BM_SRCA = BM_v1
    elif ind_mt1 == 1:
        RF_SRCA = RF_v2
        BM_SRCA = BM_v2
    elif ind_mt1 == 2:
        RF_SRCA = RF_v3
        BM_SRCA = BM_v3
    elif ind_mt1 == 3:
        RF_SRCA = RF_v4
        BM_SRCA = BM_v4

    if ind_mt2 == 0:
        RF_SRCB = RF_v1
        BM_SRCB = BM_v1
    elif ind_mt2 == 1:
        RF_SRCB = RF_v2
        BM_SRCB = BM_v2
    elif ind_mt2 == 2:
        RF_SRCB = RF_v3
        BM_SRCB = BM_v3
    elif ind_mt2 == 3:
        RF_SRCB = RF_v4
        BM_SRCB = BM_v4

    if ind_mt3 == 0:
        RF_META = RF_v1
        BM_META = BM_v1
    elif ind_mt3 == 1:
        RF_META = RF_v2
        BM_META = BM_v2
    elif ind_mt3 == 2:
        RF_META = RF_v3
        BM_META = BM_v3
    elif ind_mt3 == 3:
        RF_META = RF_v4
        BM_META = BM_v4

    ind = random.randrange(1, 8000)
    sub_ind = random.uniform(0.7, 1)
    cp_ind = random.uniform(0.5, 1)
    label_batch = np.reshape(np.transpose(label_mt[:, :, ind]), [1, 128, 128, 1])

    data1_SRCA = np.reshape(VSG_AUG(RF_SRCA[:, 3018 * 0:3018 * 1, ind], sub_ind, cp_ind),
                       [1, 256, 3018, 1])
    data2_SRCA = np.reshape(VSG_AUG(RF_SRCA[:, 3018 * 1:3018 * 2, ind], sub_ind, cp_ind),
                       [1, 256, 3018, 1])
    data3_SRCA = np.reshape(VSG_AUG(RF_SRCA[:, 3018 * 2:3018 * 3, ind], sub_ind, cp_ind),
                       [1, 256, 3018, 1])
    data4_SRCA = np.reshape(VSG_AUG(RF_SRCA[:, 3018 * 3:3018 * 4, ind], sub_ind, cp_ind),
                       [1, 256, 3018, 1])
    data5_SRCA = np.reshape(VSG_AUG(RF_SRCA[:, 3018 * 4:3018 * 5, ind], sub_ind, cp_ind),
                       [1, 256, 3018, 1])
    data6_SRCA = np.reshape(VSG_AUG(RF_SRCA[:, 3018 * 5:3018 * 6, ind], sub_ind, cp_ind),
                       [1, 256, 3018, 1])
    data7_SRCA = np.reshape(VSG_AUG(RF_SRCA[:, 3018 * 6:3018 * 7, ind], sub_ind, cp_ind),
                       [1, 256, 3018, 1])
    bmode_SRCA = np.reshape(np.transpose(BM_SRCA[:, :, ind]), [1, BM_SRCA.shape()[0], BM_SRCA.shape()[1], 1])



    sub_ind = random.uniform(0.7, 1)
    cp_ind = random.uniform(0.5, 1)
    data1_SRCB = np.reshape(VSG_AUG(RF_SRCB[:, 3018 * 0:3018 * 1, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    data2_SRCB  = np.reshape(VSG_AUG(RF_SRCB[:, 3018 * 1:3018 * 2, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    data3_SRCB  = np.reshape(VSG_AUG(RF_SRCB[:, 3018 * 2:3018 * 3, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    data4_SRCB  = np.reshape(VSG_AUG(RF_SRCB[:, 3018 * 3:3018 * 4, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    data5_SRCB  = np.reshape(VSG_AUG(RF_SRCB[:, 3018 * 4:3018 * 5, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    data6_SRCB  = np.reshape(VSG_AUG(RF_SRCB[:, 3018 * 5:3018 * 6, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    data7_SRCB  = np.reshape(VSG_AUG(RF_SRCB[:, 3018 * 6:3018 * 7, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    bmode_SRCB = np.reshape(np.transpose(BM_SRCB[:, :, ind]), [1, BM_SRCB.shape()[0], BM_SRCB.shape()[1], 1])


    sub_ind = random.uniform(0.7, 1)
    cp_ind = random.uniform(0.5, 1)
    data1_META = np.reshape(VSG_AUG(RF_META[:, 3018 * 0:3018 * 1, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    data2_META = np.reshape(VSG_AUG(RF_META[:, 3018 * 1:3018 * 2, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    data3_META = np.reshape(VSG_AUG(RF_META[:, 3018 * 2:3018 * 3, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    data4_META = np.reshape(VSG_AUG(RF_META[:, 3018 * 3:3018 * 4, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    data5_META = np.reshape(VSG_AUG(RF_META[:, 3018 * 4:3018 * 5, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    data6_META = np.reshape(VSG_AUG(RF_META[:, 3018 * 5:3018 * 6, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    data7_META = np.reshape(VSG_AUG(RF_META[:, 3018 * 6:3018 * 7, ind], sub_ind, cp_ind),
                            [1, 256, 3018, 1])
    bmode_META = np.reshape(np.transpose(BM_META[:, :, ind]), [1, BM_META.shape()[0], BM_META.shape()[1], 1])


    return label_batch, data1_SRCA,data2_SRCA,data3_SRCA,data4_SRCA,data5_SRCA,data6_SRCA,data7_SRCA, bmode_SRCA,\
           data1_SRCB,data2_SRCB,data3_SRCB,data4_SRCB,data5_SRCB,data6_SRCB,data7_SRCB, bmode_SRCB,\
           data1_META,data2_META,data3_META,data4_META,data5_META,data6_META,data7_META, bmode_META

