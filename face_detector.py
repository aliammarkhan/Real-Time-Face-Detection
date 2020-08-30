# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:44:05 2020

@author: Biohazard
"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

bbt = ['Amy','Bernadette','Howard','Leonard','Penny','Raj','Sheldon'] #classes 
model = load_model('model.h5') #loading model
f_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #for dace detection
video = cv2.VideoCapture('The.Big.Bang.Theory.S12E13.720p.AMZN.WEB-DL.x265-HETeam.mkv') #enter path of a big bang theory episode
X = [] #for saving X cord of rectangle
Y = [] #for saving Y cord of rectangle
face = [] #List to iterate for each face
prediction = None 
while(True):
    ret,frame = video.read() #read video frame by frame
    faces = f_cascade.detectMultiScale(frame,1.3,5) 
    for (x,y,w,h) in faces:
        X.append(x)
        Y.append(y)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) #draw rectangle along face
        face.append(frame[y:y+h,x:x+w])#cropping face
        
    for x,y,f in zip(X,Y,face): #for each face detected
        #processing frame
        if type(f) is np.ndarray and  not faces is ():
            f = cv2.resize(f,(224,224)) #resizing acc to model input
            f = Image.fromarray(f,'RGB')
            f = np.array(f)
            f = (f) *(1.0/255.0)
            #adding a dimension
            f = np.expand_dims(f,axis = 0)
            prediction = model.predict(f)[0] >0.5
            print(prediction)
          
        if len(np.where(prediction == True)[0])  == 1:
            cv2.putText(frame,bbt[np.where(prediction == True)[0][0]], (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame,"Not found", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2) #press q to exit
        
            
    cv2.imshow('Video',frame)
    X =[]
    Y =[]
    face = []
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()