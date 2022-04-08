import cv2 as cv
import sys
import os

from cv2 import getStructuringElement
from cv2 import MORPH_RECT
from cv2 import CAP_MSMF

vid = cv.VideoCapture(0)
path =  'images'

#Preguntar que descriptor utilizar
print('Que descriptor usar: 1. ORB  2. SIFT  3. HOG')
descriptor = 1 
if(descriptor == 1):
    descriptor = 'ORB'
elif(descriptor == 2):
    descriptor = 'SIFT'
elif(descriptor == 3):
    descriptor = 'HOG'

#Leer y procesar las imagenes de la carpeta
images = []
imgNames = []
mylist = os.listdir(path)
for i in mylist:
    #Leer la imagen
    imgCur = cv.imread(f'{path}/{i}', 0)
    #Procesar las imagenes
    kernel = getStructuringElement(MORPH_RECT,(3,3))
    imgCur = cv.erode(imgCur, kernel, iterations=1)
    imgCur = cv.dilate(imgCur, kernel, iterations=1)
    images.append(imgCur)
    imgNames.append(os.path.splitext(i)[0])

print(imgNames)

#Crear los descriptores
descriptorList = []
for img in images:
    if(descriptor == 'ORB'):
        orb = cv.ORB_create(nfeatures = 1000)
        kpRef, desRef = orb.detectAndCompute(img, None)
        descriptorList.append(desRef)
    elif(descriptor == 'SIFT'):
        print(2)
    elif(descriptor == 'HOG'):
        print(3)


while(True):
    #Leer y procesar la señal de video
    ret, vidFrame = vid.read()
    vidFrame = cv.cvtColor(vidFrame,cv.COLOR_BGR2GRAY)
    kernel = getStructuringElement(MORPH_RECT,(3,3))
    vidFrame = cv.erode(vidFrame, kernel, iterations = 1)
    vidFrame = cv.dilate(vidFrame, kernel, iterations = 1)

    idImg = -1
    #Encontrar las caracteristicas interesantes
    if(descriptor == 'ORB'):
        kpVid, desVid = orb.detectAndCompute(vidFrame, None)
        bf = cv.BFMatcher()
        matchList = []
        try:
            for des in descriptorList:
                matches = bf.knnMatch(des, desVid, k=2)
                good = []
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        good.append([m])
                matchList.append(len(good))
        except:
            pass
        if(len(matchList) != 0):
            if max(matchList) > 15:
                idImg = matchList.index(max(matchList))
            
    elif(descriptor == 'SIFT'):
        print()
    elif(descriptor == 'HOG'):
        print()

    if idImg != -1:
        cv.putText(vidFrame, imgNames[idImg], (50,50), cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

    cv.imshow('Video', vidFrame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()

  