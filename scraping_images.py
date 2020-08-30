# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 18:26:23 2020

@author: Biohazard
"""

from simple_image_download import simple_image_download as simp 
import os
import cv2

def extract_face(imagePath): #function for extracting faces from images
    img = cv2.imread(imagePath)
    f_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = f_cascade.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2) #draw a rectangle around the face
        face = img[y:y+h,x:x+w]  #crop the face
        return face

response = simp.simple_image_download

query= ["Sheldon Cooper","Leonard Hofstadter","Penny Hofstadter","Howard Wolowitz","Raj Koothrappali","Amy Farrah Fowler","Bernadette Wolowitz"]

#downloading images
no_of_images = 50 # no of images to download
for q in query:
   response().download(q, no_of_images, extensions={'.jpg', '.png', '.jpeg'})

Source = 'simple_images/'
Dest = 'Dataset/Train/'

#create the respectives directories
try:
    os.mkdir('Dataset')
    os.mkdir('Dataset/Train')
    for q in query:
        os.mkdir('Dataset/Train/' + q.split()[0])
except OSError:
    pass

separator = '_'
for q in query:
    a =separator.join(q.split()) + '/'
    data = [x for x in os.listdir(Source + a) if os.path.getsize(Source + a + x)>0] #get file names
    for train in data:
        img = extract_face(Source+a+train) #extract faces
        if img is None:
            continue
        cv2.imwrite(Dest+q.split()[0] + '/'+train, img) #save the cropped face
   

        