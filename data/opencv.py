import cv2
import os

def loadImages(path_r = "input/"):
    return [os.path.join(path_r, f) for f in os.listdir(path_r) if f.endswith(".jpg")]

filenames = loadImages()
images = []
for file in filenames:
    images.append(cv2.imread(file, cv2.IMREAD_UNCHANGED))

                  
print(images)                  

num = 0
for image in images:
    input_image = image
    img = cv2.flip(input_image, 1)
    # blur image to reduce noise
    img_blur = cv2.medianBlur(img, 5)

    # equalize the histogram to improve contrast
    img_eq = cv2.equalizeHist(img_blur)
    # create Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img_blur)

    # set global threshold value to eliminate grey values (binary)
    th0 = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret,tsh2 = cv2.threshold(th0,127,255,cv2.THRESH_BINARY_INV)

    # median to reduce noise
    median = cv2.medianBlur(tsh2, 3)
    # blur to smooth edges
    blur = cv2.GaussianBlur(median, (3, 3), 0)
    
    #output image in output folder
    path = "output/"
    cv2.imwrite(os.path.join(path, str(num)+".jpg"), blur)
    num += 1
  
  
