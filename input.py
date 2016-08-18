import numpy as np
import random
import PIL
import lxml.etree as ET
import Image
import re
import os
import cPickle
IMAGE_WIDTH = 448
IMAGE_HEIGHT = 448
CCOORD = 5
CNOOBJ = 0.5
CLASS_NUM = 20
B = 2
S = 7
GRID_SIZE = 64
classes = ['aeroplane','bicycle','bird','boat','bottle',
           'bus','car','cat','chair','cow',
           'diningtable','dog','horse','motorbike','person',
           'pottedplant','sheep','sofa','train','tvmonitor']

'''
get needed label structure from original input
params:
    c : class
    x1,y1 : left-top coordinate
    x2,y2 : right-bottom coordinate
return:
    label : an 26D array
    index 0 ~ 19 : 20 classes
    index 20, 21 : center coordinate of true box, represent by offset of grid(i,j), range 0 to 1
    index 22, 23 : true box size, width and height, normalized by image width and height, range 0 to 1
    index 24 ~ 72 : the grid responsible for object, 0 or 1
'''
def getlabel(c,x1,y1,x2,y2):
    classes = np.zeros([20])
    classes[c] = 1
    xc = (x1 + x2) * 1.0 / 2
    yc = (y1 + y2) * 1.0 / 2
    # the offset
    x_offset = (xc % GRID_SIZE) * 1.0 / GRID_SIZE
    y_offset = (yc % GRID_SIZE) * 1.0 / GRID_SIZE
    # size
    width = (x2 - x1 + 1) * 1.0 / IMAGE_WIDTH
    height = (y2 - y1 + 1) * 1.0 / IMAGE_HEIGHT
    boxes = np.array([x_offset, y_offset, width, height])
    # the resposible grid , (i,j)
    gx = int (xc / GRID_SIZE)
    gy = int (yc / GRID_SIZE)
    grids = np.zeros([7,7])
    grids[gx][gy] = 1
    cells = np.reshape(grids,[49])
    label = np.hstack((classes, boxes, cells))
    return label

'''
from label to better format
'''
def recover(label):
    classes = label[:20]
    c = np.argmax(classes)
    w = label[22] * IMAGE_WIDTH - 1
    h = label[23] * IMAGE_HEIGHT - 1
    grids = label[24:]
    index = np.argmax(grids)

    gx = index / 7
    gy = index % 7
    print gx,gy
    xc = GRID_SIZE * (label[20] + gx)
    yc = GRID_SIZE * (label[21] + gy)
    #print w,h,xc,yc
    x1 = xc - w/2
    x2 = xc + w/2
    y1 = yc - h/2
    y2 = yc + h/2
    return [c,x1,y1,x2,y2]


'''
for test
generate random cases

'''
def generate_random_cases(num):
    res = []
    for i in range(num):
        x1 = random.randint(0, IMAGE_WIDTH - 1)
        x2 = random.randint(0, IMAGE_WIDTH - 1)
        y1 = random.randint(0, IMAGE_HEIGHT - 1)
        y2 = random.randint(0, IMAGE_HEIGHT - 1)
        #c = random.randint(0, 19)
        c = num
        tmp = getlabel(c,min(x1,x2),min(y1,y2),max(x1,x2),max(x2,y2))
        res.append(tmp)
    res = np.reshape(res,[num,73])
    return res

#train = generate_random_cases(4)
##print train
'''
c = random.randint(0,19)
object_class = c
x1 = 1
y1 = 1
x2 = 133
y2 = 444
positions = [x1,y1,x2,y2]
label = getlabel(c,x1,y1,x2,y2)
print label
print recover(label)
'''

def get_data():
    data = cPickle.load(open("/home/starsea/tensorflow/yolo/data.pkl",'rb'))
    images = [ d["image"] for d in data]
    labels = [ d["label"] for d in data]
    return images,labels
