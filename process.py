# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 19:50:13 2024

@author: tanch
"""

'''
from PIL import Image
import pytesseract
import argparse
import cv2
import os
path = "temp/french-license-plate9.jpg"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--image", default=path, help="path to input image to be OCR'd")
ap.add_argument("--preprocess", type=str, default="thresh", help="type of preprocessing to be done")
args = vars(ap.parse_args())

# load the example image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 3)
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)
# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)'''

import easyocr
import cv2
import string

path = "temp/crops/plate/car.jpg"
alpha = list(string.ascii_uppercase)
nb = [str(i) for i in range(10)]


def processing(path_img):
    # load the example image and convert it to grayscale
    image = cv2.imread(path_img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, tresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0) 
    #cv2.imwrite("temp/g1.png", gray)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #gray = cv2.GaussianBlur(gray, (5, 5), 0) 
    #gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #tresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    # invert the image 
    #♦invert = cv2.bitwise_not(gray) 
    # cv2.imwrite("temp/g4.png", invert)
        
    
    # kernel = np.ones((2,2),np.uint8)
    # i = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # i = cv2.erode(gray, kernel, iterations=1) 

    cv2.imwrite("temp/test.png", gray)

    reader = easyocr.Reader(["en"])
    list_res = reader.readtext("temp/test.png")
    return list_res

def read_lp(list_res):
    if len(list_res) != 0:
        lp = list_res[0][1]
        temp = list_res[0][2]
        for l in list_res:
            if l[2]>temp:
                lp = l[1]
        #return ''.join(clean_lp(l))
        return check_lp(lp)
    return ""
        
        
        
def check_lp(lp_txt):
    l = list(lp_txt.upper())
    if len(l)!= 0:
        l_new = clean_lp(l)
        size = len(l_new)
        seq = ""
        for k in l_new:
            if k in alpha:
                seq+="0"
            elif k in nb:
                seq+="1"
            else:
                seq+="2"
        if "0011100" in seq:        # 9 9 9 A A A 9 9"
            seq = list(seq)
            for i in range(size):
                if seq[i] == "0":
                    bool_test = True
                    test = list("011100")
                    for j in range(i+1, i+7):
                        if seq[j] != test[j-(i+1)]:
                            bool_test = False
                    if bool_test:
                        list_res = [l_new[n] for n in range(i, i+7)]
                        return ''.join(list_res)
                    
        elif "11100011" in seq:     # 9 9 9 A A A 9 9"
            seq = list(seq)
            for i in range(size):
                if seq[i] == "1":
                    bool_test = True
                    test = list("1100011")
                    for j in range(i+1, i+8):
                        if seq[j] != test[j-(i+1)]:
                            bool_test = False
                    if bool_test:
                        list_res = [l_new[n] for n in range(i, i+8)]
                        return ''.join(list_res)
        return "ERROR "+''.join(clean_lp(l_new))
    return ''


def clean_lp(list_txt):
    """
    ['@', 'S', ' ', '6', '9', '5', ' ', 'H', 'G', ' ', '5']
    TO
    ['S', '6', '9', '5', 'H', 'G', '5']

    Parameters
    ----------
    list_txt : list
        list of lp read split.

    Returns
    -------
    res : list
        list clean.

    """
    res = []
    for elt in list_txt:
        if elt in alpha:
            res.append(elt)
        elif elt in nb:
            res.append(elt)
    return res


def get_center(xyxy):
    xmin, ymin, xmax, ymax = xyxy[0].tolist(), xyxy[1].tolist(), xyxy[2].tolist(), xyxy[3].tolist()
    tl, br = (xmin, ymax), (xmax, ymin)
    center = (abs(xmax-xmin), abs(ymax-ymin))
    return center

def get_distance(ptA, ptB):
    x1, x2, y1, y2 = ptA[0], ptB[0], ptA[1], ptB[1]
    distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    return distance

def track(xyxy, previous_track, i_cadre=50):
    # xmin, ymin, xmax, ymax, conf, class == xyxy
    center = get_center(xyxy)
    minim = None
    temp_dis = -1
    for k, previous_pos in previous_track.items():
        distance = get_distance(center, previous_pos)
        if temp_dis < 0:
            temp_dis = distance
        if distance <= i_cadre:
            if distance < temp_dis:
                temp_dis = distance
                minim = k
    return minim

def clean_track(dict_track, list_keys):
    res = {}
    for key in list_keys:
        res.update({key: dict_track[key]})
    return res

if __name__ == "__main__":
    res = processing(path)
    print(res)
    txt = read_lp(res)
    print(txt)
