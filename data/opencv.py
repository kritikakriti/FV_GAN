import cv2
import os

def loadImages(path_r = "input/"):
    return [os.path.join(path_r, f) for f in os.listdir(path_r) if f.endswith(".bmp")]

filenames = loadImages()
images = []
for file in filenames:
    images.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))

num = 0
for image in images:
    path = "output/"
    cv2.imwrite(os.path.join(path, str(num)+".bmp"), image)
    num += 1
  
  
