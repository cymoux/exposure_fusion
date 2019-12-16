#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from scipy.ndimage.filters import laplace
import numpy as np
import glob


# In[2]:


folder = input("Insert the folder name which contains the image to fuse (in the folder 'sample'): ")

levels = 5 #pyramid level

# BGR order
img = []
for fname in glob.glob('sample/'+folder+'/'+'*'):
    img_k = cv2.imread(fname)
    img_k = img_k[-int(2**levels*np.floor(len(img_k)/2**levels)):,-int(2**levels*np.floor(len(img_k[0])/2**levels)):,] # Cropping the picture to get a multiple of 2**levels pixels
    b,g,r = cv2.split(img_k)       # get b,g,r
    img_k = cv2.merge([r,g,b])     # convert BGR to RGB
    img.append(img_k)

img = np.array(img)

# Careful: img[0] contains image_1.jpg, and so on
img_norm = img/255

# 1 - Contrast
eps=0.0001
# We convert the normalised images in grayscale and then apply a Laplace filter
C = np.array([np.abs(laplace(cv2.cvtColor(img[i],cv2.COLOR_RGB2GRAY)/255)) for i in range(len(img))])+eps

# 2 - Saturation

S = np.array([np.std(img_norm[i],axis=2) for i in range(len(img_norm))])+eps

# 3 - Well-exposedness

std = 0.2 # standard deviation

img_blue_gauss = np.array([np.exp(-(img_norm[i,:,:,0]-0.5)**2/(2*(std**2))) for i in range (len(img_norm))])
img_green_gauss = np.array([np.exp(-(img_norm[i,:,:,1]-0.5)**2/(2*(std**2))) for i in range (len(img_norm))])
img_red_gauss = np.array([np.exp(-(img_norm[i,:,:,2]-0.5)**2/(2*(std**2))) for i in range (len(img_norm))])

E = img_blue_gauss*img_green_gauss*img_red_gauss+eps

# 4 - Weight map

W = C*S*E # weights
# 5 - Fusion

W_normalised = (np.sum(W,axis=0)**(-1))*W # normalised weights
W_BGR = np.expand_dims(W_normalised, 3) # for BGR
W_BGR = np.repeat(W_BGR, 3, axis=3) # same weights for each channel

R = np.sum(W_BGR*img_norm, axis=0) # resulting image for naive method


# In[3]:


#%% Method using Laplacian Pyramid

def gaussian_pyramid(image, index):
    layer = image.copy()
    gp = [layer]

    for i in range(index):
        layer = cv2.pyrDown(layer)
        gp.append(layer)

    return gp

def laplacian_pyramid(image, index):
    gp = gaussian_pyramid(image, index+1)

    lp = [gp[index]]

    for i in range(index, 0, -1):
        GE = cv2.pyrUp(gp[i])
        layer = cv2.subtract(gp[i-1],GE)
        lp.append(layer)
    lp.reverse()
    return lp


# In[4]:


L_I = []

for image in img_norm:
    lp = laplacian_pyramid(image, levels)
    L_I.append(lp)

# Gaussian Pyramid for weight maps
G_W = []

for weight_map in W_BGR:
    # generate Gaussian Pyramid
    gp = gaussian_pyramid(weight_map, levels)

    G_W.append(gp)

G_W = np.array(G_W)
L_R = np.sum(G_W * L_I, axis=0)

laplacian_pyramid = L_R

reconstructed_image = laplacian_pyramid[-1]

for i in range(levels - 1, -1, -1):
    size = (reconstructed_image[i].shape[1], reconstructed_image[i].shape[0])
    reconstructed_image = cv2.pyrUp(reconstructed_image)#, dstsize=size)
    reconstructed_image = cv2.add(reconstructed_image, laplacian_pyramid[i])
    
r, g, b = cv2.split(reconstructed_image)       # get r, g, b
reconstructed_image = cv2.merge([b,g,r])     # convert RGB to BGR
cv2.imwrite('sample/'+folder+'/'+folder+'_fused.png', 255*reconstructed_image)
