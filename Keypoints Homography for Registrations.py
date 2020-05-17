# -*- coding: utf-8 -*-
"""
Created on Sun May 17 03:09:35 2020

@author: thejunaidiqbal
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt

im1 = cv2.imread('images/distorted.jpg')
im2 = cv2.imread('images/monkey.jpg')  #train img

img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(50)  #50 points registers

kp1, des1 = orb.detectAndCompute(img1, None)  #find keypoints
kp2, des2 = orb.detectAndCompute(img2, None)


matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

matches = matcher.match(des1, des2, None)  

matches = sorted(matches, key = lambda x:x.distance)


img3 = cv2.drawMatches(im1,kp1, im2, kp2, matches[:10], None)

cv2.imshow("Matches image", img3)
cv2.waitKey(0)



points1 = np.zeros((len(matches), 2), dtype=np.float32)  
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
   points1[i, :] = kp1[match.queryIdx].pt    
   points2[i, :] = kp2[match.trainIdx].pt


  
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
height, width, channels = im2.shape
im1Reg = cv2.warpPerspective(im1, h, (width, height))  
   
print("Estimated homography : \n",  h)

cv2.imshow("Registered image", im1Reg)
cv2.waitKey()