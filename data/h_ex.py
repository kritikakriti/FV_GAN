"""
@donblob
Copyright(C) 2017 donblob, donblob@posteo.org

This program is free software: you can redistribute it and / or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import cv2
import sys
import os
from matplotlib import pyplot as plt

print("HandVeinExtractor v1.0")
print("Usage: python3 fve2.py <path to veinimage-input> <path to veinimage-output> <v for verbose output, "
      "blank for processing only>")

if len(sys.argv) < 3:
    print("Too few arguments. please see <usage> for more informations")
    exit()

elif len(sys.argv[1]) < 1 and not os.path.exists(sys.argv[1]):
    print(sys.argv[1] + " is not a valid file")
    exit()

elif len(sys.argv[2]) != 1 and not str(sys.argv[2]).endswith(".bmp"):
    print("<" + sys.argv[2] + ">" + " has no valid file type. Please specify a valid output file (e.g. edit.bmp)")
    exit()

elif len(sys.argv) > 4:
    print("Too many arguments. please see <usage> for more informations")

elif len(sys.argv) == 4 and sys.argv[3] != '-v':
    print("<" + sys.argv[3] + ">" + " unrecognized option")
    exit()

else:
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # load images
    input_image = cv2.imread(input_file, 0)
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

    # median to reduce noise
    median = cv2.medianBlur(th0, 3)
    # blur to smooth edges
    blur = cv2.GaussianBlur(median, (3, 3), 0)

    # save to disk
    cv2.imwrite(output_file, blur)
    print("Success! File " + output_file + " has been written.")

    # List with all images and titles
    titles = ['Original Image with CLAHE', 'adaptiveThreshold 11,3', 'Median Noise Reduction',
              'Gaussian Blur 3,3 (Output Image)']
    images = [cl1, th0, median, blur]

    # print stuff
    if len(sys.argv) == 4 and sys.argv[3] == '-v':
        for i in range(4):
            plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
