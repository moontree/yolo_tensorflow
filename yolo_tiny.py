import numpy as np
import tensorflow as tf
import Image
import ImageDraw
import random
import math
import cPickle
import time
import sys
import input
class YOLO_TF:
    IMAGE_WIDTH = 448
    IMAGE_HEIGHT = 448
    CCOORD = 5
    CNOOBJ = 0.5
    CLASS_NUM = 20
    B = 2
    S = 7
    GRID_SIZE = 64
    BATCH_SIZE = 3
    fromfile = None
    tofile_img = 'test/output.jpg'
    tofile_txt = 'test/output.txt'
    autosave = True
    imshow = True
    filewrite_img = False
    filewrite_txt = False
    disp_console = True
    smart_learn = True
    alpha = 0.1
    threshold = 0.2
    iou_threshold = 0.5
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    '''def __init__(self, argvs=[]):
        self.argv_parser(argvs)
        self.build_networks()
        if self.fromfile is not None: self.detect_from_file(self.fromfile)'''

    '''def argv_parser(self, argvs):
        for i in range(1, len(argvs), 2):
            if argvs[i] == '-fromfile': self.fromfile = argvs[i + 1]
            if argvs[i] == '-tofile_img': self.tofile_img = argvs[i + 1]; self.filewrite_img = True
            if argvs[i] == '-tofile_txt': self.tofile_txt = argvs[i + 1]; self.filewrite_txt = True
            if argvs[i] == '-imshow':
                if argvs[i + 1] == '1':
                    self.imshow = True
                else:
                    self.imshow = False
            if argvs[i] == '-disp_console':
                if argvs[i + 1] == '1':
                    self.disp_console = True
                else:
                    self.disp_console = False'''

    def build_networks(self):
        if self.disp_console: print "Building YOLO_tiny graph..."
        self.x = tf.placeholder('float32', [None, 448, 448, 3])
        self.labels = tf.placeholder('float32', [None, 73])
        self.learning_rate = tf.placeholder('float32')
        self.conv_1 = self.conv_layer(1, self.x, 16, 3, 1)
        self.pool_2 = self.pooling_layer(2, self.conv_1, 2, 2)
        self.conv_3 = self.conv_layer(3, self.pool_2, 32, 3, 1)
        self.pool_4 = self.pooling_layer(4, self.conv_3, 2, 2)
        self.conv_5 = self.conv_layer(5, self.pool_4, 64, 3, 1)
        self.pool_6 = self.pooling_layer(6, self.conv_5, 2, 2)
        self.conv_7 = self.conv_layer(7, self.pool_6, 128, 3, 1)
        self.pool_8 = self.pooling_layer(8, self.conv_7, 2, 2)
        self.conv_9 = self.conv_layer(9, self.pool_8, 256, 3, 1)
        self.pool_10 = self.pooling_layer(10, self.conv_9, 2, 2)
        self.conv_11 = self.conv_layer(11, self.pool_10, 512, 3, 1)
        self.pool_12 = self.pooling_layer(12, self.conv_11, 2, 2)
        self.conv_13 = self.conv_layer(13, self.pool_12, 1024, 3, 1)
        self.conv_14 = self.conv_layer(14, self.conv_13, 1024, 3, 1)
        self.conv_15 = self.conv_layer(15, self.conv_14, 1024, 3, 1)
        self.fc_16 = self.fc_layer(16, self.conv_15, 256, flat=True, linear=False)
        self.fc_17 = self.fc_layer(17, self.fc_16, 4096, flat=False, linear=False)
        # skip dropout_18
        self.fc_19 = self.fc_layer(19, self.fc_17, 1470, flat=False, linear=True)
        #self.labels = tf.placeholder(tf.float32, [None, 73])
        self.loss = self.cal_loss(self.fc_19, self.labels)
        #self.learning_rate = 0.001
        self.train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss)
        if self.disp_console: print "Loading complete!" + '\n'



    '''
    build a conv layer.
    params:
        idx : index
        inputs : input
        filters : feature number
        size : conv size
        stride : stride
    '''
    def conv_layer(self, idx, inputs, filters, size, stride):
        channels = inputs.get_shape()[3]
        weight = _weight_init([size, size, int(channels), filters])
        biases = _bias_init([filters])
        conv = tf.nn.conv2d(inputs, weight, strides=[1, stride, stride, 1], padding='SAME',
                            name=str(idx) + '_conv')
        conv_biased = tf.add(conv, biases, name=str(idx) + '_conv_biased')
        if self.disp_console:
            print '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' %\
                  (idx, size, size, stride, filters, int(channels))
        # leacky relu
        return tf.maximum(self.alpha * conv_biased, conv_biased, name=str(idx) + '_leaky_relu')

    '''
    build a pool_layer
    params:
        idx : index
        inputs : input
        size : pool size
        stride : stride
    '''
    def pooling_layer(self, idx, inputs, size, stride):
        if self.disp_console:
            print '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx, size, size, stride)

        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME',
                              name=str(idx) + '_pool')

    '''
    build a full connect layer
    params :
        idx : index
        inputs : input
        hiddens : feature numbers
        flat : reshape from [-1,W,H,C] to [-1,W*H*c]
        linear : with/without relu
    '''
    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False):
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_processed = tf.reshape(inputs, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        weight = _weight_init([dim, hiddens])
        biases = _bias_init([hiddens])
        if self.disp_console:
            print '    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % \
                  (idx, hiddens, int(dim), int(flat), 1 - int(linear))
        if linear:
            return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx) + '_fc')

        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        return tf.maximum(self.alpha * ip, ip, name=str(idx) + '_fc')

    def detect_from_imgMat(self, imgMat):
        s = time.time()
        inputs = np.zeros((1, 448, 448, 3), dtype='float32')
        inputs[0] = (imgMat / 255.0) * 2.0 - 1.0
        in_dict = {self.x: inputs}
        net_output = self.sess.run(self.fc_19, feed_dict=in_dict)
        self.result = self.interpret_output(net_output[0])
        self.show_results(imgMat, self.result)
        strtime = str(time.time() - s)
        if self.disp_console: print 'Elapsed time : ' + strtime + ' secs' + '\n'

    def detect_from_file(self, filename):
        if self.disp_console: print 'Detect from ' + filename
        img = Image.read(filename)
        img_resize = img.resize((448, 448))
        imgMat = np.array(img_resize)
        # img = misc.imread(filename)
        self.detect_from_cvmat(imgMat)

    def detect_from_crop_sample(self):
        self.w_img = 640
        self.h_img = 420
        f = np.array(open('person_crop.txt', 'r').readlines(), dtype='float32')
        inputs = np.zeros((1, 448, 448, 3), dtype='float32')
        for c in range(3):
            for y in range(448):
                for x in range(448):
                    inputs[0, y, x, c] = f[c * 448 * 448 + y * 448 + x]

        in_dict = {self.x: inputs}
        net_output = self.sess.run(self.fc_19, feed_dict=in_dict)
        self.boxes, self.probs = self.interpret_output(net_output[0])
        img = cv2.imread('person.jpg')
        self.show_results(self.boxes, img)

    def interpret_output(self, output):
        probs = np.zeros((7, 7, 2, 20))
        class_probs = np.reshape(output[0:980], (7, 7, 20))
        scales = np.reshape(output[980:1078], (7, 7, 2))
        boxes = np.reshape(output[1078:], (7, 7, 2, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
        boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
        boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

        boxes[:, :, :, 0] *= self.w_img
        boxes[:, :, :, 1] *= self.h_img
        boxes[:, :, :, 2] *= self.w_img
        boxes[:, :, :, 3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0: continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1],
                           boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def show_results(self, img, results):
        img_cp = Image.fromarray(img)
        if self.filewrite_txt:
            ftxt = open(self.tofile_txt, 'w')
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3]) // 2
            h = int(results[i][4]) // 2
            if self.disp_console: print '    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(
                y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4])) + '], Confidence = ' + str(
                results[i][5])
            if self.filewrite_img or self.imshow:
                draw = ImageDraw.Draw(img_cp)
                draw.line((x - w, y - h), (x - w, y + h), (x + w, y - h), (x + w, y + h))
                draw.text((x - w + 5, y - h - 7), results[i][0] + ' : %.2f' % results[i][5], fill=(255, 255, 255))
            if self.filewrite_txt:
                ftxt.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + ',' + str(
                    results[i][5]) + '\n')
        if self.filewrite_img:
            if self.disp_console: print '    image file writed : ' + self.tofile_img
            img_cp.save(self.tofile_img,'JPEG')
        if self.imshow:
            img_cp.show()
        if self.filewrite_txt:
            if self.disp_console: print '    txt file writed : ' + self.tofile_txt
            ftxt.close()

    def save_weights(self, filename):
        if self.saver :
            #self.saver = tf.train.Saver()
            self.saver.save(self.sess, filename)
            if self.disp_console:
                print 'save weights to ' + filename
        else :
            if(self.disp_console):
                print 'There is no session with tensorflow'

    def restore_weights(self, filename):
        print filename
        if self.saver :
            self.saver.restore(self.sess, filename)
            print 'restore weights from file ' + filename
        else :
            print 'There is no session with tensorflow'
            #self.sess = tf.Session()
            #self.sess.run(tf.initialize_all_variables())
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, filename)

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
    def cal_loss(self, pred, real):
        pred = tf.reshape(pred,[-1, 7, 7, 30])
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
        object_in_grids = tf.reshape(real[:,24:],[-1,7,7])
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
            #print 'responsible_box shape : ', responsible_box[:, :, :, 0].get_shape()
            #print 'temp_coord_loss shape : ', temp_coord_loss.get_shape()
            temp_coord_loss = tf.reshape(box1_loss, [-1, 7, 7, 1])
            coord_loss = self.CCOORD * tf.mul(objects_in_grids, temp_coord_loss)
            #coord_loss = self.CCOORD * temp_coord_loss
            #print 'coord_loss shape : ', coord_loss.get_shape()
        with tf.variable_scope('obj_loss') as scope:
            temp_obj_loss = tf.reduce_sum(tf.mul(responsible_box, tf.square( (objects_in_grids - pred_p_objects))),reduction_indices=3)
            temp_obj_loss = tf.reshape(temp_obj_loss, [-1, 7, 7, 1])
            obj_loss = self.CCOORD * tf.mul(objects_in_grids,temp_obj_loss)
            # print 'obj_loss shape : ', obj_loss.get_shape()
        with tf.variable_scope('noobj_loss') as scope:
            temp_noobj_loss = tf.reduce_sum(tf.mul(responsible_box, tf.square( (objects_in_grids - pred_p_objects))),
                                            reduction_indices=3)
            temp_noobj_loss = tf.reshape(temp_noobj_loss, [-1, 7, 7, 1])
            # print 'tmp shape' , noobj_loss_tmp.get_shape()
            noobj_loss = self.CNOOBJ * tf.mul(1 - objects_in_grids, temp_noobj_loss)
            print 'noobj_loss shape : ', noobj_loss.get_shape()
        with tf.variable_scope('class_loss') as scope:
            classes_diff = tf.mul(objects_in_grids, pred_classes - real_classes)
            #print 'classes_diff shape ', classes_diff
            class_loss = tf.reduce_sum(tf.square(classes_diff), reduction_indices=3)
            class_loss = tf.reshape(class_loss,[-1,7,7,1])
            #class_loss = tf.mul(objects_in_grids, tf.reduce_sum(tf.square(classes_diff),reduction_indices = 3))
            # print 'class_loss shape : ', class_loss.get_shape()
        with tf.variable_scope('total_loss') as scope:
            total_loss = coord_loss + obj_loss + noobj_loss + class_loss
            # print 'total_loss shape : ', total_loss.get_shape()
            loss = tf.reduce_mean(tf.reduce_sum(total_loss, reduction_indices=[1, 2, 3]), reduction_indices=0)
        return tf.reduce_sum(loss)


    def init_network(self):
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        self.data = input.get_data()

        tf.scalar_summary('loss',self.loss)
        self.mWriter = tf.train.SummaryWriter('tiny_log', self.sess.graph)
        self.mop = tf.merge_all_summaries()
        if self.disp_console:
            print 'weights init finished'

    def train(self):
        print 'start training ...'
        index = 0 * 2
        xx = self.data[0][index : index + 2]
        yy = self.data[1][index : index + 2]
        print yy
        for epoch in range(135):
            if self.smart_learn:
                learning_rate = _get_learning_rate_by_epoch(epoch)
            else:
                learning_rate = 0.001
            for step in range(16):
                #index = step * 2
                #xx = self.data[0][index: index + 2]
                #yy = self.data[1][index: index + 2]
                _, mstr, total_loss = self.sess.run([self.train_step, self.mop, self.loss], feed_dict = {self.learning_rate : 0.001, self.x : xx,  self.labels: yy})
                print 'training epoch = %d, step = %d, loss = %f' %(epoch, step, total_loss)
                self.mWriter.add_summary(mstr, 50 * epoch + step)
            if self.autosave:
                filename = 'weights/yolo-tiny-epoch-' + str(epoch) + '.ckpt'
                self.save_weights(filename)

    def predict(self,image):
        im = Image.open(image)
        im = im.resize((self.IMAGE_WIDTH,self.IMAGE_HEIGHT))
        imgMat = np.array(im)
        res = self.sess.run(self.fc_19, feed_dict = {self.x : [imgMat]})
        res = np.reshape(res,[7,7,30])
        boxes = []
        print "i,j,c,p,confidence,x,y,w,h'"
        for i in range(7):
            for j in range(7):
                c = np.argmax(res[i][j][:20])
                responsible_box = 0
                #if res[i][j][28] < res[i][j][29]:
                    #responsible_box = 1
                if i == 2 and j == 4:
                    boxes.append([i, j, c, res[i][j][c], res[i][j][28 + responsible_box],
                                  res[i][j][20 + 4 * responsible_box], res[i][j][21 + 4 * responsible_box],
                                  res[i][j][22 + 4 * responsible_box], res[i][j][23 + 4 * responsible_box]])
                if i == 3 and j == 3:
                    boxes.append([i, j, c, res[i][j][c], res[i][j][28 + responsible_box],
                                  res[i][j][20 + 4 * responsible_box], res[i][j][21 + 4 * responsible_box],
                                  res[i][j][22 + 4 * responsible_box], res[i][j][23 + 4 * responsible_box]])
                '''if(res[i][j][c] > -2.0):
                    score_th = 0.5
                    responsible_box = 0
                    if res[i][j][28] < res[i][j][29]:
                        responsible_box = 1
                    if res[i][j][28 + responsible_box] > score_th:
                        w = res[i][j][22 + 4 * responsible_box]
                        h = res[i][j][23 + 4 * responsible_box]
                        size_threshold = -3.00
                        if w > size_threshold and h > size_threshold:
                            boxes.append([i,j,c,res[i][j][c],res[i][j][28+responsible_box],
                                      res[i][j][20 + 4 * responsible_box],res[i][j][21 + 4 * responsible_box],
                                      res[i][j][22 + 4 * responsible_box],res[i][j][23 + 4 * responsible_box]])'''


        print boxes

        draw = ImageDraw.Draw(im)
        for box in boxes :
            w = box[7] * self.IMAGE_WIDTH / 2
            h = box[8] * self.IMAGE_HEIGHT / 2
            print 'w = ', w, ' h = ', h
            lx = (box[0] + box[5]) * self.GRID_SIZE - w
            ly = (box[1] + box[6]) * self.GRID_SIZE - h
            tx = (box[0] + box[5]) * self.GRID_SIZE + w
            ty = (box[1] + box[6]) * self.GRID_SIZE + h
            print(lx,ly,tx,ty)
            draw.rectangle((lx,ly,tx,ty))
            content = self.classes[box[2]] + ' p = ' + str(box[3])
            draw.text((lx, ly), content, fill=(255, 255, 255))
        im.show()






def _weight_init(shape):
    print 'shape = ' ,shape
    stddev = 2.0
    for a in shape[:]:
        stddev /= a
    stddev = math.sqrt(stddev)
    var = tf.truncated_normal(shape, stddev = stddev)
    return tf.Variable(var)

def _bias_init(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def _get_learning_rate_by_epoch(epoch):
    if epoch < 75 :
        return 0.001
    else:
        if epoch < 105:
            return 0.0001
        else:
            return 0.00001

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

if __name__ == '__main__':
    yolo_tiny = YOLO_TF()
    yolo_tiny.build_networks()
    yolo_tiny.init_network()
    yolo_tiny.restore_weights("/home/starsea/tensorflow/yolo/weights/yolo-tiny-epoch-13.ckpt")
    yolo_tiny.predict('/home/starsea/data/VOC2007/JPEGImages/000001.jpg')
    #yolo_tiny.train()