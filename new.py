from os import listdir
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from os.path import isdir
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import cv2
from mtcnn.mtcnn import MTCNN
import datetime
import sqlite3
import sys
import FunctionFull as FF

# Define
folderOfSource = FF.processed_folder
folderOfTest = FF.test_folder
folderOfRaw = FF.raw_folder
folderOfInput = FF.input_folder
NumOfSource = 30
cameraSource = 0

# Nhập ID, name
ID = FF.MakeDataProcessed()


# Tạo thư mục tương ứng, nếu đã tồn tạo thì tạo đè lên
FF.MkFolderOfID(ID, folderOfSource)

# Xác định khuôn mặt và lưu vào folder đã tạo
FF.FaceOfTrainMTCNNcam(ID, cameraSource, NumOfSource)


# Trích xuất đặc trưng và lưu vector embedding
PathOfEmbTrain = FF.processed_folder + str(ID) 
resultnew = FF.LoadFaceGetVectoEmbedded(PathOfEmbTrain, str(ID))
# print('Make Folder of ID = {} done, 1'.format(str(ID)))

# Kiểm tra 1 tập dữ liệu với các tập dữ liệu còn lại
folderAll = folderOfSource
FolderEmbTest = folderOfSource + str(ID) + '/' + str(ID) + '-txt'
isyou = []
avereach = []
maxeach = []
avertemp = 0
resulteach = []
maxcp = 0
amaxcp = 0
for file in listdir(FolderEmbTest):
    fileOfTest = file
    ReadEmbTest = FF.ReadEmbVector(folderOfSource, file, ID)
    for file in listdir(folderAll):
        IDsource = str(file)
        if IDsource == str(ID):
            pass
        else:
            FolderEmbTrain = folderAll + IDsource + '/' + IDsource + '-txt'
            for file in listdir(FolderEmbTrain):
                ReadEmbTrain = FF.ReadEmbVector(folderOfSource, file, file.split('-')[0])
                result = abs((FF.CompareTwoEmbVector(ReadEmbTest, ReadEmbTrain))*100)
                # print('Compare Test {} & Train {} = {}'.format(fileOfTest, file, result))
                avertemp = avertemp + result
                resulteach.append(result)
            avertemp = float(avertemp/len(resulteach))
            maxcp = max(maxcp, max(resulteach))
            amaxcp = max(amaxcp, avertemp)
            avertemp = 0
            resulteach = []
if (int(maxcp) <= 50) & (int(amaxcp) <= 40):
    print('Most matched is {}(%) and largest average is {}(%), 1'.format(maxcp, amaxcp))
    print('collect dataset successfully')
else: 
    print('Most matched with orther people is {}(%) and largest average is {}(%)-Please try again for better, 0'.format(maxcp, amaxcp))
    print('the dataset is not good enough')










