import tensorflow as tf
import numpy as np
import random
import math
import input
import cPickle
IMAGE_WIDTH = 448
IMAGE_HEIGHT = 448
CCOORD = 5
CNOOBJ = 0.5
CLASS_NUM = 20
B = 2
S = 7
GRID_SIZE = 64
BATCH_SIZE = 3

#init
def weight_init(shape, name):
    stddev = 2.0
    for a in shape[:-1]:
        stddev /= a
    stddev = math.sqrt(stddev)
    var = tf.truncated_normal(shape, stddev = stddev)
    return tf.Variable(var)

def bias_init(shape, name):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)
#conv
def conv2d(x, W, stride = 1):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = 'SAME')
#maxpool
def maxpool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


'''
build conv layer
params:
    input : a W * H * D matrix
    conv : conv size [S,S,C]
    conv_stride : conv stride
    scopename : scopename
return :
    W * H * C matrix
'''
def conv_layer(input,conv,conv_stride,scopename):
    _, width, height, depth = [d.value for d in input.get_shape()]
    [cwidth,cheight,cdepth] = conv
    #print [cwidth,cheight,depth,cdepth]
    with tf.variable_scope(scopename) as scope:
        weight = weight_init([cwidth,cheight,depth,cdepth], name = 'W_' + scopename)
        bias = bias_init([cdepth], name = 'B_' + scopename)
        local = tf.nn.bias_add(conv2d(input,weight,conv_stride),bias)
        relu_result = prelu(local)
    return relu_result

'''
many conv layers with/without maxpool
params :
    input : W * H * D
    convs : an array of conv size
    stride : conv stride [[S,S,C]]
    scopename :
    pool : 1 means do maxpool, 0 means not
return :
    W' * H' * C
'''
def conv_layers_with_maxpool(input,convs,stride,scopename,pool = 1):
    with tf.variable_scope(scopename) as scope:
        xx = input
        for i in range(len(convs)):
            xx = conv_layer(xx, convs[i], stride, scopename + '_' + str(i))
        if pool == 1:
            local = maxpool(xx)
        else:
            local = xx
    return local

'''
calculate ious in matrix form
params :
bx is the predict boxes
by is the true box
calculate in form of matrix
return a 1-D array
'''
def cal_iou(bx,by):
    tb1 = tf.minimum(bx[ : , : , : , 0] + 0.5 * bx[ : , : , : , 2], by[ : , : , : , 0] + 0.5 * by[ : , : , : , 2]) - \
          tf.maximum(bx[ : , : , : , 0] - 0.5 * bx[ : , : , : , 2], by[ : , : , : , 0] - 0.5 * by[ : , : , : , 2])
    lr1 = tf.minimum(bx[ : , : , : , 1] + 0.5 * bx[ : , : , : , 3], by[ : , : , : , 1] + 0.5 * by[ : , : , : , 3]) - \
          tf.maximum(bx[ : , : , : , 1] - 0.5 * bx[ : , : , : , 3], by[ : , : , : , 1] - 0.5 * by[ : , : , : , 3])
    tb = tf.maximum(tb1, tb1 * 0)
    lr = tf.maximum(lr1, lr1 * 0)
    res2 = tf.mul(tb, lr)
    res = tf.div(res2, tf.sub(tf.mul(bx[ : , : , : , 2], bx[ : , : , : , 3]) +
                              tf.mul(by[ : , : , : , 2], by[ : , : , : , 3]), res2))
    return res

'''
leaky relu :
return:
    x, if x > 0
    0.1 * x, if x < 0
'''
def prelu(x):
    xz = 0.1 * x
    return tf.maximum(x, xz)


'''
box1 : ? * 7 * 7 * 4
box2 : ? * 7 * 7 * 4
real_box : ? * 4
return :
     ? * 7 * 7 * 2
'''
def cal_ious(box1, box2, real_box):
    IOUS_1 = tf.reshape(cal_iou(box1, real_box),[-1, 7, 7, 1])
    IOUS_2 = tf.reshape(cal_iou(box2, real_box), [-1, 7, 7, 1])
    res = tf.concat(3, [IOUS_1, IOUS_2])
    print 'IOUS_SHAPE : ', res.get_shape()
    return res




'''
calculate total loss
----------predict result----------
    shape : ?  * 7 * 7 * 30
    0~19 : classes
    20~23 : box1
    24~27 : box2
    28,29 : p_object
----------real result----------
    shape ?  * 20 + 4 + 49
    0 ~ 19 : classes
    20 ~ 23 : true box
    24 ~ 72 : cell has object or not
'''
def cal_loss(pred,real):
    pred_classes = pred[:, :, :, :20]
    pred_boxes_1 = pred[:, :, :, 20:24]
    pred_boxes_2 = pred[:, :, :, 24:28]
    pred_p_objects = pred[:, :, :, 28:29]
    real_classes = tf.reshape(real[:, :20], [-1, 1, 1, 20])
    real_boxes = tf.reshape(real[:, 20:24], [-1, 1, 1, 4])
    #print 'real_boxes shape', real_boxes.get_shape()
    objects_in_grids = tf.reshape(real[:, 24:], [-1, 7, 7, 1])

    # IOUS shape :  ? * 7 * 7 * 2
    with tf.variable_scope('IOUS') as scope:
        IOUS = cal_ious(pred_boxes_1, pred_boxes_2, real_boxes)

    with tf.variable_scope('responsible_box') as scope:
        IOUS_max = tf.cast(tf.reshape(tf.greater(IOUS[:, :, :, 0],IOUS[:, :, :, 1]), [-1, 7, 7, 1]), tf.float32)
        IOUS_min = tf.cast(tf.reshape(tf.less(IOUS[:, :, :, 0], IOUS[:, :, :, 1]), [-1, 7, 7, 1]), tf.float32)
        responsible_box = tf.concat(3, [IOUS_max, IOUS_min])
    # print 'responsible_box shape : ' , responsible_box.get_shape()
    with tf.variable_scope('center_loss') as scope:
        box1_loss = tf.square(pred_boxes_1[:, :, :, 0] - real_boxes[:, :, :, 0]) + \
                    tf.square(pred_boxes_1[:, :, :, 1] - real_boxes[:, :, :, 1]) + \
                    tf.square(pred_boxes_1[:, :, :, 2] - real_boxes[:, :, :, 2]) + \
                    tf.square(pred_boxes_1[:, :, :, 3] - real_boxes[:, :, :, 3])
        # tf.square(tf.sqrt(pred_boxes_1_w[:, :, :]) - tf.sqrt(real_boxes[:, :, :, 2])) + \
        # tf.square(tf.sqrt(pred_boxes_1_h[:, :, :]) - tf.sqrt(real_boxes[:, :, :, 3]))

    with tf.variable_scope('size_loss') as scope:
        box2_loss = tf.square(pred_boxes_2[:, :, :, 0] - real_boxes[:, :, :, 0]) + \
                    tf.square(pred_boxes_2[:, :, :, 1] - real_boxes[:, :, :, 1]) + \
                    tf.square(pred_boxes_2[:, :, :, 2] - real_boxes[:, :, :, 2]) + \
                    tf.square(pred_boxes_2[:, :, :, 3] - real_boxes[:, :, :, 3])
        # tf.square(tf.sqrt(pred_boxes_2_w[:, :, :]) - tf.sqrt(real_boxes[:, :, :, 2])) + \
        # tf.square(tf.sqrt(pred_boxes_2_h[:, :, :]) - tf.sqrt(real_boxes[:, :, :, 3]))

    with tf.variable_scope('coord_loss') as scope:
        temp_coord_loss = tf.mul(responsible_box[:, :, :, 0], box1_loss) + tf.mul(responsible_box[:, :, :, 1],
                                                                                  box2_loss)
        print 'responsible_box shape : ', responsible_box[:, :, :, 0].get_shape()
        print 'temp_coord_loss shape : ', temp_coord_loss.get_shape()
        temp_coord_loss = tf.reshape(temp_coord_loss, [-1, 7, 7, 1])
        coord_loss = CCOORD * tf.mul(objects_in_grids, temp_coord_loss)
        print 'coord_loss shape : ', coord_loss.get_shape()
    with tf.variable_scope('obj_loss') as scope:
        obj_loss = tf.mul(objects_in_grids,
                          tf.reduce_sum(tf.mul(responsible_box, tf.square(IOUS * (1 - pred_p_objects)))))
        # print 'obj_loss shape : ', obj_loss.get_shape()
    with tf.variable_scope('noobj_loss') as scope:
        temp_noobj_loss = tf.reduce_sum(tf.mul(responsible_box, tf.square(IOUS * (1 - pred_p_objects))),
                                        reduction_indices=3)
        temp_noobj_loss = tf.reshape(temp_noobj_loss, [-1, 7, 7, 1])
        # print 'tmp shape' , noobj_loss_tmp.get_shape()
        noobj_loss = CNOOBJ * tf.mul(1 - objects_in_grids, temp_noobj_loss)
        # print 'noobj_loss shape : ', noobj_loss.get_shape()
    with tf.variable_scope('class_loss') as scope:
        classes_diff = pred_classes - real_classes
        # print 'classes_diff shape ', classes_diff
        class_loss = tf.mul(objects_in_grids, tf.reduce_sum(tf.square(classes_diff)))
        # print 'class_loss shape : ', class_loss.get_shape()
    with tf.variable_scope('total_loss') as scope:
        total_loss = coord_loss + obj_loss + noobj_loss + class_loss
        # print 'total_loss shape : ', total_loss.get_shape()
        loss = tf.reduce_mean(tf.reduce_sum(total_loss, reduction_indices=[1, 2, 3]), reduction_indices=0)
    return loss

'''
YOLO net architecture
'''
def conv_net(x):
    layer = x
    layer1_convs = [[5, 5, 64]]
    layer1 = conv_layers_with_maxpool(layer, layer1_convs, 2, 'CONV1')
    layer2_convs = [[3, 3, 192]]
    layer2 = conv_layers_with_maxpool(layer1, layer2_convs, 1, 'CONV2')
    layer3_convs = [[1, 1, 128],[3, 3, 256],[1, 1, 256],[3, 3, 512]]
    layer3 = conv_layers_with_maxpool(layer2, layer3_convs, 1, 'CONV3')
    layer4_convs = [[1,1,256],[3,3,512],[1,1,256],[3,3,512],[1, 1, 256],[3,3,512],[1,1,256],[3,3,512],[1,1,512],[3,3,1024]]
    layer4 = conv_layers_with_maxpool(layer3, layer4_convs , 1, 'CONV4')
    temp_convs = [[1, 1, 512],[3, 3, 1024],[1, 1, 512],[3, 3, 1024]]
    temp_layer = conv_layers_with_maxpool(layer4, temp_convs, 1, 'TEMP_CONV',0)
    layer5_conv1 = conv_layer(temp_layer, [3, 3, 1024], 1, 'CONV5_1')
    layer5_conv2 = conv_layer(layer5_conv1, [3, 3, 1024], 2, 'CONV5_2')
    layer6_convs = [[3, 3, 1024], [3, 3, 1024]]
    layer6 = conv_layers_with_maxpool(layer5_conv2, layer6_convs, 1, 'CONV6',0)

    with tf.variable_scope('FC1') as scope:
        _, w, h, d = [i.value for i in layer6.get_shape()]
        hn = tf.reshape(layer6,[-1, w * h * d])
        weight = weight_init([w * h * d, 4096], name = 'W_FC1')
        bias = bias_init([4096], name = 'B_FC1')
        layer7 = prelu(tf.nn.bias_add(tf.matmul(hn, weight), bias))

    with tf.variable_scope('FC2') as scope:
        weight = weight_init([4096, 7 * 7 * 30], name = 'W_FC2')
        #bias = bias_init([7 * 7 * 30], name = 'B_FC2')
        bias = tf.Variable(tf.constant(0.0, shape = [7 * 7 * 30]),name = 'B_FC2')
        layer8 = prelu(tf.nn.bias_add(tf.matmul(layer7, weight), bias))
        #tf.histogram_summary('bias',bias)
        output = tf.reshape(layer8,[-1, 7, 7, 30])
    return output

'''
generate learning rate by epochs
'''
def get_learning_rate(epoch):
    if epoch < 75 :
        return 0.01
    else:
        if epoch < 105:
            return 0.001
        else :
            return 0.0001

# build network

image = tf.placeholder(tf.float32,[None, 448, 448, 3])
labels = tf.placeholder(tf.float32,[None, 73])
y = conv_net(image)
with tf.variable_scope('loss') as scope:
    #loss = cal_loss(y,labels)
    pred = y
    real = labels
    pred_classes = pred[:, :, :, :20]
    pred_boxes_1 = pred[:, :, :, 20:24]
    pred_boxes_2 = pred[:, :, :, 24:28]
    pred_p_objects = pred[:, :, :, 28:29]
    real_classes = tf.reshape(real[:, :20], [-1, 1, 1, 20])
    real_boxes = tf.reshape(real[:, 20:24], [-1, 1, 1, 4])
    # print 'real_boxes shape', real_boxes.get_shape()
    objects_in_grids = tf.reshape(real[:, 24:], [-1, 7, 7, 1])

    # IOUS shape :  ? * 7 * 7 * 2
    with tf.variable_scope('IOUS') as scope:
        IOUS = cal_ious(pred_boxes_1, pred_boxes_2, real_boxes)

    with tf.variable_scope('responsible_box') as scope:
        IOUS_max = tf.cast(tf.reshape(tf.greater(IOUS[:, :, :, 0], IOUS[:, :, :, 1]), [-1, 7, 7, 1]), tf.float32)
        IOUS_min = tf.cast(tf.reshape(tf.less_equal(IOUS[:, :, :, 0], IOUS[:, :, :, 1]), [-1, 7, 7, 1]), tf.float32)
        responsible_box = tf.concat(3, [IOUS_max, IOUS_min])
    # print 'responsible_box shape : ' , responsible_box.get_shape()
    with tf.variable_scope('center_loss') as scope:
        box1_loss = tf.square(pred_boxes_1[:, :, :, 0] - real_boxes[:, :, :, 0]) + \
                    tf.square(pred_boxes_1[:, :, :, 1] - real_boxes[:, :, :, 1]) + \
                    tf.square(pred_boxes_1[:, :, :, 2] - real_boxes[:, :, :, 2]) + \
                    tf.square(pred_boxes_1[:, :, :, 3] - real_boxes[:, :, :, 3])
        # tf.square(tf.sqrt(pred_boxes_1_w[:, :, :]) - tf.sqrt(real_boxes[:, :, :, 2])) + \
        # tf.square(tf.sqrt(pred_boxes_1_h[:, :, :]) - tf.sqrt(real_boxes[:, :, :, 3]))

    with tf.variable_scope('size_loss') as scope:
        box2_loss = tf.square(pred_boxes_2[:, :, :, 0] - real_boxes[:, :, :, 0]) + \
                    tf.square(pred_boxes_2[:, :, :, 1] - real_boxes[:, :, :, 1]) + \
                    tf.square(pred_boxes_2[:, :, :, 2] - real_boxes[:, :, :, 2]) + \
                    tf.square(pred_boxes_2[:, :, :, 3] - real_boxes[:, :, :, 3])
        # tf.square(tf.sqrt(pred_boxes_2_w[:, :, :]) - tf.sqrt(real_boxes[:, :, :, 2])) + \
        # tf.square(tf.sqrt(pred_boxes_2_h[:, :, :]) - tf.sqrt(real_boxes[:, :, :, 3]))

    with tf.variable_scope('coord_loss') as scope:
        temp_coord_loss = tf.mul(responsible_box[:, :, :, 0], box1_loss) + tf.mul(responsible_box[:, :, :, 1],
                                                                                  box2_loss)
        print 'responsible_box shape : ', responsible_box[:, :, :, 0].get_shape()
        print 'temp_coord_loss shape : ', temp_coord_loss.get_shape()
        temp_coord_loss = tf.reshape(temp_coord_loss, [-1, 7, 7, 1])
        coord_loss = CCOORD * tf.mul(objects_in_grids, temp_coord_loss)
        print 'coord_loss shape : ', coord_loss.get_shape()
    with tf.variable_scope('obj_loss') as scope:
        obj_loss = tf.mul(objects_in_grids,
                          tf.reduce_sum(tf.mul(responsible_box, tf.square(IOUS * (1 - pred_p_objects)))))
        # print 'obj_loss shape : ', obj_loss.get_shape()
    with tf.variable_scope('noobj_loss') as scope:
        temp_noobj_loss = tf.reduce_sum(tf.mul(responsible_box, tf.square(IOUS * (1 - pred_p_objects))),
                                        reduction_indices=3)
        temp_noobj_loss = tf.reshape(temp_noobj_loss, [-1, 7, 7, 1])
        # print 'tmp shape' , noobj_loss_tmp.get_shape()
        noobj_loss = CNOOBJ * tf.mul(1 - objects_in_grids, temp_noobj_loss)
        # print 'noobj_loss shape : ', noobj_loss.get_shape()
    with tf.variable_scope('class_loss') as scope:
        classes_diff = pred_classes - real_classes
        # print 'classes_diff shape ', classes_diff
        class_loss = tf.mul(objects_in_grids, tf.reduce_sum(tf.square(classes_diff)))
        # print 'class_loss shape : ', class_loss.get_shape()
    with tf.variable_scope('total_loss') as scope:
        total_loss = coord_loss + obj_loss + noobj_loss + class_loss
        # print 'total_loss shape : ', total_loss.get_shape()
        loss = tf.reduce_mean(tf.reduce_sum(total_loss, reduction_indices=[1, 2, 3]), reduction_indices=0)
    tf.scalar_summary('loss',loss)

learning_rate = tf.placeholder(tf.float32)
train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)


test_x, test_y = input.get_data()
#def train_network(net):
sess = tf.Session()
mWriter = tf.train.SummaryWriter('log', sess.graph)
mop = tf.merge_all_summaries()
sess.run(tf.initialize_all_variables())
for epoch in range(135):
    #lr = get_learning_rate(epoch)
    #print 'lr = ',lr
    #weights_file = 'model.ckpt'
    #print 'weights_file = ', weights_file
    #print tf.train_variables()
    for i in range(16):
        index = i * 2
        xx = test_x[i:i+1]
        yy = test_y[i:i + 1]
        _,t1,mstr,rb = sess.run([train_step, loss, mop,box1_loss], feed_dict = {learning_rate : 0.000001, image : xx, labels : yy})
        print 'epoch : ', epoch, ' step = ', i, ' loss = ', t1
        print rb[0][0][0]
        mWriter.add_summary(mstr, 50 * epoch + i)
        #if i == 0:
            #saver.save(sess, weights_file)'''

def restore_network(net,weights_file):
    sess = tf.Session()
    saver = tf.train.Saver()
    return saver.restore(sess,weights_file)





#train_network(y)
