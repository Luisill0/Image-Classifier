import cv2 as cv
import sys

from cv2 import getStructuringElement
from cv2 import MORPH_RECT

vid = cv.VideoCapture(0)

#Leer y procesar la imagen de referencia
imgRef = cv.imread("smg.jpg",0)
kernel = getStructuringElement(MORPH_RECT,(3,3))
imgRef = cv.erode(imgRef, kernel, iterations = 1)
imgRef = cv.dilate(imgRef, kernel, iterations = 1)

while(True):
    #Leer y procesar la señal de video
    ret, vidFrame = vid.read()
    vidFrame = cv.cvtColor(vidFrame,cv.COLOR_BGR2GRAY)
    kernel = getStructuringElement(MORPH_RECT,(3,3))
    vidFrame = cv.erode(vidFrame, kernel, iterations = 1)
    vidFrame = cv.dilate(vidFrame, kernel, iterations = 1)
    
    #ORB detector
    orb = cv.ORB_create(nfeatures = 1000)
    kp1, des1 = orb.detectAndCompute(imgRef,None)
    kp2, des2 = orb.detectAndCompute(vidFrame,None)

    imgKp1 = cv.drawKeypoints(imgRef,kp1,None)
    imgKp2 = cv.drawKeypoints(vidFrame,kp2,None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    img3 = cv.drawMatchesKnn(imgRef, kp1, vidFrame, kp2, good, None, flags=2)

    cv.imshow('Kp1', imgKp1)
    cv.imshow('Kp2', imgKp2)
    cv.imshow('Knn', img3)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()

cv.imshow('KeyPoints', img3)
print(len(good))
cv.waitKey(0)
cv.destroyAllWindows()
#https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/