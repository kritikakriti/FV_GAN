import cv2
import os

def loadImages(path = "input/"):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".bmp")]



filenames = loadImages()
images = []
for file in filenames:
    images.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
#cv2.imread()  
  
  
