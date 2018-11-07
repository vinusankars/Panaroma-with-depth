#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:37:56 2018

@author: vinusankars
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def crop(img, threshold=0):
    if len(img.shape) == 3:
        flat = np.max(img, 2)
    else:
        flat = img
    assert len(flat.shape) == 2

    rows = np.where(np.max(flat, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flat, 1) > threshold)[0]
        img = img[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        img = img[:1, :1]

    return img

def matchCompute(img1, img2, draw=0, n_match=10):
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    temp = bf.match(des1, des2)
    match = sorted(temp, key = lambda x:x.distance)
    
    if draw == 1:
        img3 = cv.drawMatches(img1, kp1, img2, kp2, match[:n_match], None, flags=6)
        plt.figure(figsize=(20,20))
        plt.imshow(cv.cvtColor(np.array(img3, dtype='uint8'), cv.COLOR_BGR2RGB))
        plt.show()
        
    return match, kp1, des1, kp2, des2
    

def homography(mm1, mm2):
    H, maxInlier = [] , []
    print('\nCalculating homography...')
    for K in range(1000):
        rand1 = np.random.randint(0, len(mm1))
        (y1, x1) = mm1[rand1]
        (y11, x11) = mm2[rand1]
        
        rand2 = np.random.randint(0, len(mm1))
        (y2, x2) = mm1[rand2]
        (y21, x21) = mm2[rand2]
        
        rand3 = np.random.randint(0, len(mm1))
        (y3, x3) = mm1[rand3]
        (y31, x31) = mm2[rand3]
        
        rand4 = np.random.randint(0, len(mm1))
        (y4, x4) = mm1[rand4]
        (y41, x41) = mm2[rand4]      
          
        mat = []
        mat.append([x1, y1, 1, 0, 0, 0, -x11*x1, -x11*y1, -x11])
        mat.append([0, 0, 0, x1, y1, 1, -y11*x1, -y11*y1, -y11])
        
        mat.append([x2, y2, 1, 0, 0, 0, -x21*x2, -x21*y2, -x21])
        mat.append([0, 0, 0, x2, y2, 1, -y21*x2, -y21*y2, -y21])
        
        mat.append([x3, y3, 1, 0, 0, 0, -x31*x3, -x31*y3, -x31])
        mat.append([0, 0, 0, x3, y3, 1, -y31*x3, -y31*y3, -y31])
        
        mat.append([x4, y4, 1, 0, 0, 0, -x41*x4, -x41*y4, -x41])
        mat.append([0, 0, 0, x4, y4, 1, -y41*x4, -y41*y4, -y41])
        
        mat = np.matrix(mat)
        u, s, v = np.linalg.svd(mat)
        h = np.reshape(v[8], (3,3))
        h = h/h.item(8)
        
        inlier = []
        for i in range(len(mm1)):
            (n1, m1) = mm1[i]
            (n2, m2) = mm2[i]            
            
            p1 = np.transpose(np.matrix([m1, n1, 1]))
            estp2 = np.dot(h, p1)
            estp2 = estp2/estp2.item(2)
            
            p2 = np.transpose(np.matrix([m2, n2, 1]))
            error = p2-estp2
            sigma = 2
            if np.linalg.norm(error) < sigma*(5.99)**0.5:
                inlier.append([m1, n1, m2, n2])
                
        if len(inlier) > len(maxInlier):
            maxInlier = inlier
            H = h
            
#        threshold = 0.5
#        if len(maxInlier) > threshold*len(match):
#            break
    print(H)
    return H

def warp(img1, img2, h, ksize):
    h = np.linalg.inv(h)
    h = h/h.item(2)
#    mat1 = np.transpose(np.dot(h, np.transpose(np.matrix([1, 0, 1]))))
#    mat1 = mat1/mat1.item(2)
#    mat2 = np.transpose(np.dot(h, np.transpose(np.matrix([1, img1.shape[1], 1]))))
#    mat2 = mat2/mat2.item(2)
    x_off = 500
    y_off = img1.shape[1]*2#max(int(abs(mat1.item(1))), int(abs(mat2.item(1))))
#    print(mat1, mat2)
    im = np.zeros((img1.shape[0]*3, img1.shape[1]*5, 3))
    
    print('\nTransforming...')
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            im[i+x_off][j+y_off] = img1[i][j]
                
#    plt.figure(figsize=(15,15))
#    plt.imshow(cv.cvtColor(np.array(im, dtype='uint8'), cv.COLOR_BGR2RGB))
#    plt.show()
    print('\nStitching...')
    
    for i in range(len(img2)):
        for j in range(len(img2[0])):
            if list(img2[i][j]) != [0,0,0]:
                mat = np.transpose(np.dot(h, np.transpose(np.matrix([i, j, 1]))))
                mat = mat/mat.item(2)
                try:
                    xx = int(mat.item(0))
                    yy = int(mat.item(1))
                    if xx+x_off-ksize>=0 and yy+y_off-ksize>=0:
                        if list(im[xx+x_off][yy+y_off]) == [0,0,0]:
                            im[xx+x_off-ksize:xx+x_off+ksize+1, yy+y_off-ksize:yy+y_off+ksize+1] = np.zeros((ksize*2+1,ksize*2+1,3), dtype='uint8')+img2[i][j]
                except:
                    continue
    
    print('\nBlending...')
    for i in range(10, len(im)-10):
        try:
            im[i][y_off] = np.average(np.average(im[i-10:i+11, y_off-10:y_off+11], 1),0).astype('uint8')
        except: 
            continue
    im = crop(im)
    plt.figure(figsize=(15,15))
    plt.imshow(cv.cvtColor(np.array(im, dtype='uint8'), cv.COLOR_BGR2RGB))
    plt.show()
    
    return im

def homo(img1, img2, thresh=5, K=1000):
    H, maxInlier = [] , []
    match, kp1, d1, kp2, d2 = matchCompute(img1, img2)
    print('\nCalculating homography...')
    for K in range(1000):
        rand1 = np.random.randint(0, len(match))
        dmatch = match[rand1]
        i1 = dmatch.queryIdx
        i2 = dmatch.trainIdx
        (y1, x1) = kp1[i1].pt
        (y11, x11) = kp2[i2].pt
        
        rand2 = np.random.randint(0, len(match))
        dmatch = match[rand2]
        i1 = dmatch.queryIdx
        i2 = dmatch.trainIdx
        (y2, x2) = kp1[i1].pt
        (y21, x21) = kp2[i2].pt
        
        rand3 = np.random.randint(0, len(match))
        dmatch = match[rand3]
        i1 = dmatch.queryIdx
        i2 = dmatch.trainIdx
        (y3, x3) = kp1[i1].pt
        (y31, x31) = kp2[i2].pt
        
        rand4 = np.random.randint(0, len(match))
        dmatch = match[rand4]
        i1 = dmatch.queryIdx
        i2 = dmatch.trainIdx
        (y4, x4) = kp1[i1].pt
        (y41, x41) = kp2[i2].pt        
          
        mat = []
        mat.append([x1, y1, 1, 0, 0, 0, -x11*x1, -x11*y1, -x11])
        mat.append([0, 0, 0, x1, y1, 1, -y11*x1, -y11*y1, -y11])
        
        mat.append([x2, y2, 1, 0, 0, 0, -x21*x2, -x21*y2, -x21])
        mat.append([0, 0, 0, x2, y2, 1, -y21*x2, -y21*y2, -y21])
        
        mat.append([x3, y3, 1, 0, 0, 0, -x31*x3, -x31*y3, -x31])
        mat.append([0, 0, 0, x3, y3, 1, -y31*x3, -y31*y3, -y31])
        
        mat.append([x4, y4, 1, 0, 0, 0, -x41*x4, -x41*y4, -x41])
        mat.append([0, 0, 0, x4, y4, 1, -y41*x4, -y41*y4, -y41])
        
        mat = np.matrix(mat)
        u, s, v = np.linalg.svd(mat)
        h = np.reshape(v[8], (3,3))
        h = h/h.item(8)
        
        inlier = []
        for i in match:
            (n1, m1) = kp1[i.queryIdx].pt
            (n2, m2) = kp2[i.trainIdx].pt             
            
            p1 = np.transpose(np.matrix([m1, n1, 1]))
            estp2 = np.dot(h, p1)
            estp2 = estp2/estp2.item(2)
            
            p2 = np.transpose(np.matrix([m2, n2, 1]))
            error = p2-estp2
            sigma = thresh
            if np.linalg.norm(error) < sigma*(5.99)**0.5:
                inlier.append([m1, n1, m2, n2])
                
        if len(inlier) > len(maxInlier):
            maxInlier = inlier
            H = h
            
#        threshold = 0.6
#        if len(maxInlier) > threshold*len(match):
#            break
    print(H)
    print(len(maxInlier),len(match))
    return H

dp = cv.imread('depth_0.jpg', 0)
img1 = cv.imread('im_0.jpg')
img0 = cv.imread('im_1.jpg')

levels = 5
for i in range(dp.shape[0]):
    for j in range(dp.shape[1]):
        dp[i][j] = (255/levels)*int(dp[i][j]/(255/levels))
            
plt.imshow(dp)
plt.show()

match, k1, d1, k2, d2 = matchCompute(img0, img1)
dic1, dic2 = {}, {}

for m in match:
    i1 = m.queryIdx
    i2 = m.trainIdx
    y1, x1 = k1[i1].pt
    y2, x2 = k2[i2].pt
    level = int(dp[int(x1)][int(y1)]/51)
    try:
        dic1[level].append([x1,y1])
    except:
        dic1[level] = []
        dic1[level].append([x1,y1])
    try:
        dic2[level].append([x2,y2])
    except:
        dic2[level] = []
        dic2[level].append([x2,y2])
 
h = {}     
for i in range(levels):
    try:
        m1 = dic1[i]
        m2 = dic2[i]
        h[i] = homography(m1, m2)
        if len(m1)<5:
            del(dic1[i])
            del(dic2[i])
            del(h[i])
    except:
        continue

im = np.zeros((1500,1500,3),dtype='uint8')    
for i in range(len(img1)):
    for j in range(len(img1[0])):
        im[i+500, j+500] = img0[i,j]
        level = int(dp[i][j]/51)
        try:
            H = h[level]
        except:
            H = h[list(h.keys())[0]]
        print(H)
        m = np.dot(H, np.matrix([i,j,1]).T)
        m = m/m.item(2)
        x = int(m.item(0))
        y = int(m.item(1))
        for rr in range(-1,2):
            for cc in range(-1,2):
                try:
                    im[x+500+rr,y+500+cc] = img1[i,j]
                except:
                    continue
        
im = crop(im)
plt.figure(figsize=(15,15))
plt.imshow(im)
plt.show()

h12 = homo(img0, img1)
img1 = warp(img0, img1, h12, 2).astype('uint8')
img1 = crop(img1)
plt.figure(figsize=(15,15))
plt.imshow(img1)
plt.show()