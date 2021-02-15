from os import listdir
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from os.path import isdir
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import cv2
# from mtcnn.mtcnn import MTCNN
import datetime
import sqlite3
import sys
import FunctionFull as FF

# Define
folderOfSource = FF.processed_folder
folderOfTest = FF.test_folder
folderOfRaw = FF.raw_folder
folderOfInput = FF.input_folder
date = str(datetime.datetime.now())
dateNow = date.split(' ')[0]

    # Lấy ID từ socket
    # ID = sys.argv[1]
ID = input('ID need to verify = ')

    # Tiến hành chụp hình để xác thưc chấm công
FF.MkFolderOfID(str(ID), folderOfTest)

# FF.InputFromCapture(ID)
pathImageCompare = folderOfInput + str(ID)
FF.InputFromCapture1(ID, pathImageCompare)

    # Trích xuất đặc trưng và lưu vector embedding
PathOfEmbTest = FF.test_folder + str(ID)
FF.LoadFaceGetVectoEmbedded(PathOfEmbTest, str(ID))

    # Lấy folder compare
FolderEmbSource = folderOfSource + str(ID) + '/' + str(ID) + '-txt'
    # FolderEmbInput = folderOfTest + str(ID) + '/' + str(ID) + '-txt'
fileInput = str(ID) + '-test-' + dateNow + '.txt'
    # Tiến hành xác thực
isyou = []
avereach = []
avertemp = 0
resulteach = []
maxv = 0
minv = 0

ReadEmbTest = FF.ReadEmbVector(folderOfTest, fileInput, ID)
for file in listdir(FolderEmbSource):
    ReadEmbTrain = FF.ReadEmbVector(folderOfSource, file, ID)
    result = abs((FF.CompareTwoEmbVector(ReadEmbTest, ReadEmbTrain))*100)
        # print('Compare Test {} & Train {} = {}'.format(fileInput, file, result))
    avertemp = avertemp + result
    resulteach.append(result)

    # Tính độ chính xác trung bình, độ trùng khớp lớn nhất, nhỏ nhất
avertemp = float(avertemp/len(resulteach))
maxv = max(resulteach)
minv = min(resulteach)

pathShow = 'dataTest/TestProcessed/' + str(ID) + '/' + str(ID) + '-image' + '/' + str(ID) + '-test-' + dateNow + '.jpg'
if os.path.isfile(pathShow) == True:
    # Xác thực
    if (minv >= 40) & (maxv >= 70):
        print('Verify successfully: ID = {}\nMax matched = {}\nMin matched = {}'.format(ID,maxv,minv))
        print('Success')
    else:
        print('Verify unsuccessfully: ID = {}\nMax matched = {}\nMin matched = {}'.format(ID,maxv,minv))
        print('Unsuccess')
    #pathShow = 'dataTest/TestProcessed/' + str(ID) + '/' + str(ID) + '-image' + '/' + str(ID) + '-test-' + dateNow + '.jpg'
    #'dataTest/TestProcessed/' + str(ID) + '/' + str(ID) + '-image' + '/' + str(ID) + '-test-' + dateNow + '.jpg'
    img = cv2.imread(pathShow)
    cv2.imshow('fdf', img)
    cv2.waitKey()
else:
    print('Not exist test file, pls check again')






