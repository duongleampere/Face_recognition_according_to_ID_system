from os import listdir
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from os.path import isdir
from PIL import Image
# import pickle
from mtcnn.mtcnn import MTCNN
# from glob import glob
# import shutil
# from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import joblib
import cv2
from mtcnn.mtcnn import MTCNN
import datetime
import sqlite3
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
from PIL import Image
from PIL import ImageEnhance

processed_folder ="dataSource/processed/"
test_folder = "dataTest/TestProcessed/"
raw_folder = "dataSource/dataRaw/"
input_folder = "dataTest/DataInputCapture/"
facenet_model = load_model('facenet_keras.h5',compile=False)
face_model = load_model('face-none-1.h5')
detector = MTCNN()
dest_size = (160,160)
isfile = os.path.isfile
join = os.path.join

# Cập nhật dữ liệu trong database
def InsertOrUpdate(id):
    conn=sqlite3.connect("FaceBase.db")
    cursor=conn.execute('SELECT * FROM People WHERE ID='+str(id))
    isRecordExist=0
    for row in cursor:
        isRecordExist = 1
        break
    if isRecordExist==1:
        cmd="INSERT INTO people(ID) Values("+str(id)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()

# Hàm nhập id
def input_id():
    id = input('Nhập mã nhân viên: ')
    if id.isdigit() == True:
        if int(id)<100000000:
            id = str(id)
            # id = id.left(8,'0')
        else:
            print('Vui lòng nhập id từ 00000000 đến 99999999\n')
            id = None
    else:
        print('Vui lòng nhập id là kí tự số\n')
        id = None
    return id

# Hàm nhập name
# def input_name():
#     name = input('Nhập tên nhân viên: ')
#     if any(char.isdigit() for char in name) == False: #ko có số trong chữ
#         if len(name) < 32:
#             pass
#         else:
#             name = None
#             print('Vui lòng nhập tên không dài hơn 32 kí tự')
#     else:
#         name = None
#         print('Vui lòng không nhập số và kí tự đặc biệt')
#     return name

# Make Data Processed
def MakeDataProcessed():
    ID = input_id()
    # name = input_name()
    return ID
    InsertOrUpdate(ID)

# Tạo thư mục lớn cho 1 ID
def MkFolderOfID(id, folder):
    PathFolder = folder + str(id)
    PathImage = PathFolder + '/' + str(id) + '-image'
    PathEmbTxt = PathFolder + '/' + str(id) + '-txt'
# Kiểm tra xem folder ID đã tồn tại hay chưa??
    if os.path.exists (PathFolder):
        pass
    else:
        os.mkdir(PathFolder)
# print('Folder {} exist'.format(id))

# Kiếm tra folder ID/ID-image tồn tại hay chưa??
    if os.path.exists(PathImage):
        pass
    else:
        os.mkdir(PathImage)
# print('Folder {}-image exist'.format(id))

# Kiểm tra folder ID/ID-txt tồn tạo hay chưa??
    if os.path.exists(PathEmbTxt):
        pass
    else:
        os.mkdir(PathEmbTxt)
# print('Folder {}-txt exist'.format(str(id)))

# Chọn chế độ lưu đè hay không
def CheckForSaveMTCNN(id, NumOfTrain):
    PathFolder = processed_folder + str(id)
    PathImage = PathFolder + '/' + str(id) + '-image'
    directory = PathImage
    number_of_files = sum(1 for item in os.listdir(directory) if isfile(join(directory, item)))
    YourChoose = input('Lưu đè lên tệp cũ(1), không lưu đè lên(khác): ')
    if YourChoose == '1':
        num = 1
        NumOfTrain = NumOfTrain
    else:
        num = number_of_files + 1
        NumOfTrain = NumOfTrain + number_of_files
    return num, NumOfTrain

# Vừa lưu, vừa xác định khuôn mặt, vừa resize về (160,160)-------------------------------------------------------------
# Ảnh chỉ có 1 mặt
# Lấy dữ liệu tệp data processed từ camera
def FaceOfTrainMTCNNcam(id, camera, NumOfSource):
    date = str(datetime.datetime.now())
    dateNow = date.split(' ')[0]
    PathFolder = processed_folder + str(id)
    PathImage = PathFolder + '/' + str(id) + '-image'
    cam = cv2.VideoCapture(camera)
    num = 1
    while(True):
        ret, img = cam.read()
        img1 = cv2.flip(img, 1)
        img = cv2.flip(img, 1)
        result = detector.detect_faces(img)
        cv2.rectangle(img1,  # img
                      (181, 402),
                      (475, 114),
                      (0, 0, 255),
                      2)
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            cv2.rectangle(img1, #img
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,155,255),
                  2)
            imgSave = img[bounding_box[1]:(bounding_box[1]+bounding_box[3]), #img1
                    bounding_box[0]:(bounding_box[0]+bounding_box[2])]
            destSize = (160, 160)
            imgSave = cv2.resize(imgSave, destSize)
            imgSave1 = np.expand_dims(imgSave, axis=0)
            if person['confidence'] >= 0.5:
                pathSave = PathImage + '/' + str(id) + '-source-' + dateNow +'-' +  str(num) + '.jpg'
                cv2.imwrite(pathSave, imgSave)
                num = num + 1
        cv2.putText(img1, "so anh : " + str(num), (181, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('FaceDetectionUsingMTCNN', img1) #img
        if cv2.waitKey(NumOfSource) & 0xFF == ord('q'):
            break
        elif num > NumOfSource:
            break
    cam.release()
    cv2.destroyAllWindows()
    # print('Detection Done from camera\nReady for dataSource/processed-{}'.format(id))

# Lấy dữ liệu tệp data processed từ hình ảnh
def FaceOfTrainMTCNNreadimg(id, PathFolder):
    date = str(datetime.datetime.now())
    dateNow = date.split(' ')[0]
    PathFolderImg = PathFolder + str(id)
    PathImage = processed_folder + str(id) + '/' + str(id) + '-image'
    num = 1
    for file in listdir(PathFolderImg):
        img = cv2.imread(PathFolderImg + '/' + file)
        result = detector.detect_faces(img)
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            cv2.rectangle(img,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
            imgSave = img[bounding_box[1]:(bounding_box[1]+bounding_box[3]),
                    bounding_box[0]:(bounding_box[0]+bounding_box[2])]
            destSize = (160, 160)
            imgSave = cv2.resize(imgSave, destSize)
            # imgSave = cv2.cvtColor(imgSave, cv2.COLOR_BGR2GRAY)
            pathSave = PathImage + '/' + str(id) + '-source-' + dateNow + '-' + str(num) + '.jpg'
            cv2.imwrite(pathSave, imgSave)
        num = num + 1
        cv2.imshow('FaceDetectionUsingMTCNNreadimg', img)
    cv2.destroyAllWindows()
    # print('Detection Done from capture\nReady for dataSource/processed-{}'.format(PathFolderImg))

# Xác định khuôn mặt và trích vector đặc trưng để tiến hành so sánh
# Từ camera
def InputFromCamera(id, camera, NumOfTest):
    PathFolder = test_folder + str(id)
    PathImage = PathFolder + '/' + str(id) + '-image'
    # Lưu ảnh test vào folder datatest/ID để đánh giá
    cam = cv2.VideoCapture(camera)
    num = 1
    while(True):
        ret, img = cam.read()
        img = cv2.flip(img,1)
        result = detector.detect_faces(img)

        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            cv2.rectangle(img,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,155,255),
                  2)
            imgCompare = img[bounding_box[1]:(bounding_box[1]+bounding_box[3]),
                    bounding_box[0]:(bounding_box[0]+bounding_box[2])]
            destSize = (160, 160)
            imgCompare = cv2.resize(imgCompare, destSize)
            # imgSave = cv2.cvtColor(imgSave, cv2.COLOR_BGR2GRAY)
            pathSave = PathImage + '/' + str(id) + '-' + 'test' + str(num) + '.jpg'
            cv2.imwrite(pathSave, imgCompare)
        num = num + 1
        cv2.imshow('FaceOfTest', img)
        if cv2.waitKey(NumOfTest) & 0xFF == ord('q'):
            break
        elif num > NumOfTest:
            break
    cam.release()
    cv2.destroyAllWindows()   
    # print('Detection Done - from camera\nReady for dataTest/TestProcessed - {}'.format(id))

# Từ Capture (file ảnh có sẵn)
def InputFromCapture(id):
    PathImage = test_folder + str(id) + '/' + str(id) + '-image'
    date = str(datetime.datetime.now())
    dateNow = date.split(' ')[0]
    # Lưu ảnh test vào folder datatest/ID để đánh giá
    cam = cv2.VideoCapture(0)
    while(1):
        ret, img = cam.read() # lấy 1 ảnh duy nhất
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # l_b = np.array([0, 39, 0])
        # u_b = np.array([34, 255, 175])
        # mask = cv2.inRange(hsv, l_b, u_b)
        # res = cv2.bitwise_and(img, img, mask=mask)
        img = cv2.flip(img, 1)
        result = detector.detect_faces(img) #img
        cv2.rectangle(img,  # img
                      (181, 402),
                      (475, 114),
                      (0, 0, 255),
                      2)
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            cv2.rectangle(img, #img
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,155,255),
                  2)
            imgCompare = img[bounding_box[1]:(bounding_box[1]+bounding_box[3]), #img
                    bounding_box[0]:(bounding_box[0]+bounding_box[2])]
            destSize = (160, 160)
            imgCompare = cv2.resize(imgCompare, destSize)
        # imgSave = cv2.cvtColor(imgSave, cv2.COLOR_BGR2GRAY)
            pathSave = PathImage + '/' + str(id) + '-test-' + dateNow + '.jpg'
            # cv2.imwrite(pathSave, imgCompare)
        cv2.imshow('fff',img) #img
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if person['confidence'] >= 0.82:
                cv2.imwrite(pathSave, imgCompare)
                print("save sucessfully")
            break
    cam.release()
    cv2.destroyAllWindows()

# Từ Capture (file ảnh có sẵn)
def InputFromCapture1(id, folder):
    PathFolder = folder
    date = str(datetime.datetime.now())
    dateNow = date.split(' ')[0]
    PathImage = test_folder + str(id) + '/' + str(id) + '-image'
    # Lưu ảnh test vào folder datatest/ID để đánh giá
    num = 1
    for file in listdir(PathFolder):
        img = cv2.imread(PathFolder + '/' + file)
        result = detector.detect_faces(img)
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            cv2.rectangle(img,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,155,255),
                  2)
            imgCompare = img[bounding_box[1]:(bounding_box[1]+bounding_box[3]),
                    bounding_box[0]:(bounding_box[0]+bounding_box[2])]
            #if (307200 / (imgCompare.shape[0] * imgCompare.shape[1])) < 100:
            destSize = (160, 160)
            imgCompare = cv2.resize(imgCompare, destSize)
            # imgCompare1 = np.expand_dims(imgCompare, axis=0)
            # print(face_model.predict(imgCompare1)[0][0])
            if person['confidence'] >= 0.82:
                # imgSave = cv2.cvtColor(imgSave, cv2.COLOR_BGR2GRAY)

                pathSave = PathImage + '/' + str(id) + '-test-' + dateNow +'.jpg' #PathImage + '/' + str(id) + '-test-'+ dateNow +'.jpg
                cv2.imwrite(pathSave, imgCompare)
                num = num + 1
        cv2.imshow('FaceOfTest', img)
    cv2.destroyAllWindows()
    # print('Detection Done - from Capture\nReady for dataTest/TestProcessed - {}'.format(PathImage))

# Trích xuất cevto đặc trưng
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
# standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
# transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
# make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

# Lấy vector đặng trung
def LoadFaceGetVectoEmbedded(PathFolder, id):
    train_folder = PathFolder + '/' + str(id) + '-image'
# load từng ảnh trong folder ra và lấy vector đặc trưng
    for file in listdir(train_folder):
        img = Image.open(train_folder + '/' + file)
        pixels = np.asarray(img)
        FaceSamples = np.array(pixels)
        FaceSamples_emb = get_embedding(facenet_model, FaceSamples)
        FaceSamples_emb = np.array(FaceSamples_emb)
        SaveEmbVectorOfImage(file[0:-4], PathFolder, FaceSamples_emb, id) #Lưu vector vào file .txt

# Lưu từng vector vào từng file .txt
# Sửa ở đây 17:00
def SaveEmbVectorOfImage(filename, folder, EmbVector, id):
    FileEmbTxt = folder +'/' + str(id) + '-txt' + '/' + filename + '.txt'
    i = 0
    FileEmbTxt_handle = open(FileEmbTxt, 'w+')
    for i in range(len(EmbVector)):
        Emb = EmbVector[i]
        Emb = str(Emb)
        FileEmbTxt_handle.write(Emb)
        FileEmbTxt_handle.write(',')
    FileEmbTxt_handle.close()
    
# Đọc vector đặc trưng của các ảnh lưu trong ID.txt
def ReadEmbVector(folder, file, id):
    FolderEmbVector = folder + str(id) + '/' + str(id) + '-txt'
    FileEmbVector_handle = open(FolderEmbVector + '/' + file, 'r')
    x = FileEmbVector_handle.read()
    Emb = []
    i = 0
    y = x.split(',')
    for i in range(len(y)-1):
        z = float(y[i])
        z = round(z, 3)
        Emb = np.append(Emb, z)
        i = i + 1
    return Emb

# Hàm so sánh, đầu vào là 2 vector đặc trưng
def CompareTwoEmbVector(EmbTest, EmbTrain):
    u = EmbTest
    v = EmbTrain
    nu = norm(u)
    nv = norm(v)
    dotuv = dot(u,v)
    cosuv = dotuv/(nu*nv)
    return cosuv

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def get_vector(image):
    try:
        pixels = np.asarray(image)
        FaceSamples = np.array(pixels)
        FaceSamples_emb = get_embedding(facenet_model, FaceSamples)
        FaceSamples_emb = np.array(FaceSamples_emb)
        return FaceSamples_emb
    except:
        pass
