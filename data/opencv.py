import cv2 
import os
import numpy as np
from skimage import morphology

def loadImages(path_r = "input/"):
    return [os.path.join(path_r, f) for f in os.listdir(path_r) if f.endswith(".jpg")]

filenames = loadImages()
images = []
for file in filenames:
    images.append(cv2.imread(file, cv2.IMREAD_UNCHANGED))

                  
print(images)                  

num = 0
for image in images:
    img = image
    
    #Grayscale normalization
    norm_img = np.zeros((800,800))
    final_img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
    # blur image to reduce noise
    img_blur = cv2.medianBlur(final_img, 5)

    # equalize the histogram to improve contrast
    img_eq = cv2.equalizeHist(img_blur)
    # create Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img_blur)

    # set global threshold value to eliminate grey values (binary)
    th0 = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret,tsh2 = cv2.threshold(th0,127,255,cv2.THRESH_BINARY_INV)
    
    #denoising
    openkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(tsh2, cv2.MORPH_OPEN, openkernel)
    closedkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closedkernel)
    result = cv2.medianBlur(closed, 5)

    #thining
    pix = result
    pix = pix / 255
    pix = morphology.skeletonize(pix)
    pix = pix.astype(int) * 255
    #cv2_imshow(result)

    #clean small objects
    pix = pix / 255
    pix = pix.astype(bool)
    pix = morphology.remove_small_objects(pix, min_size=20, connectivity=2)
    pix = pix.astype(int) * 255
    
    #output image in output folder
    path = "output/"
    cv2.imwrite(os.path.join(path, str(num)+".jpg"), pix)
    num += 1
  
  
