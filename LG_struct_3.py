import scipy.io
import os
import glob
import numpy as np
import random

train_path = "./datasets/240513-room1/db7/"
test_path = "./datasets/240513-room1/query/"

LG_Struct = dict()
qCount = 0
dbCount = 0
gt_dict = dict()


# for i,fold in enumerate(os.listdir(train_path)):
# for i, fold in enumerate(os.listdir(test_path)):

a=1
# train_image_list = os.listdir(train_path)
# test_image_list = os.listdir(test_path)
train_image_list = glob.glob(f"{train_path}*.jpg")
test_image_list = glob.glob(f"{test_path}*.jpg")
# if not os.path.isdir(test_path + fold): continue
# train_image_list = os.listdir(train_path + fold)
# if not os.path.isdir(test_path + fold):
#     test_image_list = []
# else:
#     test_image_list = os.listdir(test_path + fold)

# gt_dict[fold] = [dbCount + num for num in range(len(train_image_list))]

# dbImage저장
for image in train_image_list:

    if 'dbImage' in LG_Struct:
        LG_Struct['dbImage'] = np.vstack([LG_Struct['dbImage'], np.array([image])])
        dbCount += 1

    else:
        LG_Struct['dbImage'] = np.array([image])
        dbCount += 1

# QImage저장
for image in test_image_list:

    if 'qImage' in LG_Struct:
        LG_Struct['qImage'] = np.vstack([LG_Struct['qImage'], np.array(image)])

    else:
        LG_Struct['qImage'] = np.array([image])


gt = []
# for i in range(len(LG_Struct['qImage'] )):
#     key = str(LG_Struct['qImage'][i][0]).split('/')[0] # key: group name
#     gt.append(np.array(gt_dict[key]))

LG_Struct['whichSet'] = 'Test'
LG_Struct['numDb'] = len(LG_Struct['dbImage'])
LG_Struct['numQ'] = len(LG_Struct['qImage'])
LG_Struct['posDistThr'] = 25
LG_Struct['posDistSqThr'] = 625
LG_Struct['nonTrivPosDistSqThr'] = 100
LG_Struct['gt']= gt
scipy.io.savemat('LG_Struct/240513-room1.db7.v1.mat',LG_Struct)
