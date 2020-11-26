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
    cv2.imwrite("output/", (str(num)+".png", image))
    num += 1
  
  
