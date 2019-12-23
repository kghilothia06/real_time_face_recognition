# recognize faces using calssification algo

#1. load the training data (numpy arrays of all persons)
    # x-values are stored in numoy arrays
    # y-values we need to assign for each person
#1.read a video stream using opencv
#2.extract faces out of it
#4.use knn to find prediction of face(int)
#5.map the predicted id to name ofuser
#6.display predictions on screen - bounding box and name

import numpy as np
import cv2
import os    
import matplotlib.pyplot as plt
import pandas as pd


#knn part
def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
    dist = []
    
    for i in range(train.shape[0]):
        #get the vector and label
        ix = train[i,:-1]
        iy = train[i,-1]
        #compute the distance from test point
        d = distance(test,ix)
        dist.append([d,iy])
    #sort based on distacne and get top k
    dk = sorted(dist,key=lambda x:x[0])[:k]
    #retrieve only the labels
    labels = np.array(dk)[:,-1]
    
    #get frequencies of each label 
    output = np.unique(labels,return_counts=True)
    #find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]
        
    

#initializing camera 
cap = cv2.VideoCapture(0)

#face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
face_data = []  # x values of training data
labels = [] # y value of training data
class_id = 0 #labels for given file
names = {} #mapping btw id and name

dataset_path = './data/'

#Data preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        print("loaded" + fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)
        
        #create labels for the class
        target = class_id*np.ones((data_item.shape[0]))
        class_id += 1
        labels.append(target)
        
face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

train_set = np.concatenate((face_dataset,face_labels),axis=1)
print(train_set.shape)

#testing part
file_name = input("enter the person name whose face is being scanned:")
while True:
    ret,frame = cap.read()

    #if frame is not captured for any reason then try it again
    if ret == False:
        continue
        
    faces = face_cascade.detectMultiScale(frame,1.3,5)
   
    for face in faces :
        x,y,w,h=face
        #extract (crop out reqd face) region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
    
        #predicted label (out)
        out = knn(train_set,face_section.flatten())
        
        #display on screen the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    
    cv2.imshow("faces",frame)
    key_pressed = cv2.waitKey(1) & 0XFF
    if key_pressed == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()    