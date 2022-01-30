import numpy as np
import cv2
import os
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
import math
from PIL import Image





def find_dominant_color(image):
        #Resizing parameters
        width, height = 150,150
        image = image.resize((width, height),resample = 0)
        #Get colors from image object
        pixels = image.getcolors(width * height)
        #Sort them by count number(first element of tuple)
        sorted_pixels = sorted(pixels, key=lambda t: t[0])
        #Get the most frequent color
        dominant_color = sorted_pixels[-1][1]
        return dominant_color

def preprocess_img(img, imgSize):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]]) 
        print("Image None!")

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC) # INTER_CUBIC interpolation best approximate the pixels image
                                                               # see this https://stackoverflow.com/a/57503843/7338066
    most_freq_pixel=find_dominant_color(Image.fromarray(img))
    target = np.ones([ht, wt]) * most_freq_pixel  
    target[0:newSize[1], 0:newSize[0]] = img

    img = target

    return img

def pad_img(img):
    old_h,old_w=img.shape[0],img.shape[1]

    #Pad the height.

    #If height is less than 512 then pad to 512
    if old_h<512:
        to_pad=np.ones((512-old_h,old_w))*255
        img=np.concatenate((img,to_pad))
        new_height=512
    else:
    #If height >512 then pad to nearest 10.
        to_pad=np.ones((roundup(old_h)-old_h,old_w))*255
        img=np.concatenate((img,to_pad))
        new_height=roundup(old_h)

    #Pad the width.
    if old_w<512:
        to_pad=np.ones((new_height,512-old_w))*255
        img=np.concatenate((img,to_pad),axis=1)
        new_width=512
    else:
        to_pad=np.ones((new_height,roundup(old_w)-old_w))*255
        img=np.concatenate((img,to_pad),axis=1)
        new_width=roundup(old_w)-old_w
    return img

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

def unet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs,conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    

    if(pretrained_weights):
      model.load_weights(pretrained_weights)

    return model

model=unet()
model.load_weights('./word_seg_model.h5')



def sort_word(wordlist):
    wordlist.sort(key=lambda x:x[0])
    return wordlist



def segment_into_words(line_img,idx):
    """This function takes in the line image and line index returns word images and the reference
    of line they belong to."""
    img=pad_img(line_img)
    ori_img=img.copy()
    #ori_img=np.stack((ori_img,)*3, axis=-1)
    ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
    
    img=cv2.resize(img,(512,512))
    img=np.expand_dims(img,axis=-1)
    img=img/255
    img=np.expand_dims(img,axis=0)
    seg_pred=model.predict(img)
    seg_pred=np.squeeze(np.squeeze(seg_pred,axis=0),axis=-1)
    seg_pred=cv2.normalize(src=seg_pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.threshold(seg_pred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,seg_pred)
    contours, hier = cv2.findContours(seg_pred, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    (H, W) = ori_img.shape[:2]
    (newW, newH) = (512, 512)
    rW = W / float(newW)
    rH = H / float(newH)

    coordinates=[]

    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a white rectangle to visualize the bounding rect
        # cv2.rectangle(ori_img, (int(x*rW), int(y*rH)), (int((x+w)*rW),int((y+h)*rH)), (255,0,0), 1)
        coordinates.append((int(x*rW),int(y*rH),int((x+w)*rW),int((y+h)*rH)))

    coordinates=sort_word(coordinates)  #Sorting according to x-coordinates.
    word_counter=0

    word_array=[]
    line_indicator=[]

    for (x1,y1,x2,y2) in coordinates:
        word_img=ori_img[y1:y2,x1:x2]
        word_img=preprocess_img(word_img,(128,32))
        word_img=np.expand_dims(word_img,axis=-1)
        word_array.append(word_img)
        line_indicator.append(idx)

    return line_indicator,word_array
    








