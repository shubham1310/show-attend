from __future__ import division, print_function, absolute_import
import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import os
import cv2
import matplotlib.pyplot as plt


path ='/home/msarkar/DynamicAttention/data/'
bbox = np.load(path + 'trainRandomTR_BBxs.npy')
images= np.load(path + 'trainRandomTRs.npy', mmap_mode='r')

def chooseN_BBox(bbox,padN=6):
    n = len(bbox)
    bboxes = []
    indexes = []
    for i in range(n):
        tmp=bbox[i]
        if len(tmp)!=1:
            continue;
        indexes+=[i]
        # tmp =[np.array([0.0,0.0,0.0,0.0])] +tmp
        while not(len(tmp)==padN):
            tmp.append(np.array([-1.0,-1.0,-1.0,-1.0]))
        bboxes.append(tmp)
    return np.array(bboxes),indexes

def drawBB(im,bb):
    canvas=im.copy()
    hz,wz=canvas.shape
    [y,x,h,w]=bb
    if y>=hz or x>=wz or y+h>=hz or x+w>=wz:
        return im;
    if y<0 or x<0 or y+h<0 or x+w<0:
        return im;
    canvas[y,x:x+w]=1
    canvas[y+h,x:x+w]=1
    canvas[y:y+h,x]=1
    canvas[y:y+h,x+w]=1
    return canvas

def bn_arg_scope(layers=[slim.conv2d],weight_decay=0.00004,
                            use_batch_norm=True,
                            batch_norm_decay=0.9997,
                            batch_norm_epsilon=0.001):
        """Defines the default arg scope for inception models.
         Args:
          weight_decay: The weight decay to use for regularizing the model.
          use_batch_norm: "If `True`, batch_norm is applied after each convolution.
          batch_norm_decay: Decay for batch norm moving average.
          batch_norm_epsilon: Small float added to variance to avoid dividing by zero
          in batch norm.
          Returns:
          An `arg_scope` to use for the inception models.
        """
        batch_norm_params = {
          # Decay for the moving averages.
          'decay': batch_norm_decay,
          # epsilon to prevent 0s in variance.
          'epsilon': batch_norm_epsilon,
          # collection containing update_ops.
          'updates_collections': tf.GraphKeys.UPDATE_OPS,
          }
        if use_batch_norm:
            normalizer_fn = slim.batch_norm
            normalizer_params = batch_norm_params
        else:
            normalizer_fn = None
            normalizer_params = {}
          # Set weight_decay for weights in Conv and FC layers.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope(
                layers,
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params) as sc:
                  return sc



def build_encoder(x_image,is_trainingPH,weight_decay=0.0005):
    with tf.variable_scope('encoder') as scope:
        with slim.arg_scope(bn_arg_scope()):
            with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_trainingPH):  
                with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(weight_decay),
                                              biases_initializer=tf.constant_initializer(0.1),
                                              activation_fn=tf.nn.relu):    
                    net = slim.conv2d(x_image, 32, kernel_size=[3, 3], scope='conv1')
                    net = slim.conv2d(net, 64, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')

                    net = slim.conv2d(net, 128, [3, 3], scope='conv3')
                    net = slim.conv2d(net, 128, [2, 2], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')

#                     net = slim.conv2d(net, 128, [3, 3], scope='conv5')
#                     net = slim.conv2d(net, 256, [3, 3], scope='conv6')
#                     net = slim.max_pool2d(net, [2, 2], scope='pool3')
#                     [bz,hz,wz,cz]=h_pool3.get_shape().as_list()
#                     resh = tf.reshape(h_pool3,[-1,hz*wz,cz])
                    return net
def build_decoder(featmap,is_trainingPH,weight_decay=0.0005):
    with tf.variable_scope('decoder') as scope:
        with slim.arg_scope(bn_arg_scope()):  
            with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_trainingPH):
                with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(weight_decay),
                                                  biases_initializer=tf.constant_initializer(0.1),
                                                  activation_fn=tf.nn.relu):   
                    net = featmap
#                     net = slim.conv2d_transpose(featmap, 128, [3,3], [2,2] , scope = 'deconvtran1') #print dectran1.get_shape()
#                     net = slim.conv2d(net, 128, [3, 3], scope='deconv1') #print deconv1.get_shape()
#                     net = slim.conv2d(net, 128, [3, 3], scope='deconv2') #print deconv2.get_shape()

                    net = slim.conv2d_transpose(net, 128, [3,3], [2,2] , scope = 'deconvtran2')#print dectran2.get_shape()
                    net = slim.conv2d(net, 128, [3, 3], scope='deconv3')#print deconv3.get_shape()
                    net = slim.conv2d(net, 64, [3, 3], scope='deconv4')# print deconv4.get_shape()

                    net = slim.conv2d_transpose(net, 64, [3,3], [2,2] , scope = 'deconvtran3')#print dectran3.get_shape()
                    net = slim.conv2d(net, 32, [3, 3], scope='deconv5')#print deconv5.get_shape()
                    net = slim.conv2d(net, 1, [3, 3], scope='deconv6')#print deconv6.get_shape()
#                     resh2 = slim.flatten(deconv6)
                    return net
def predictBBox(featMap,is_trainingPH):
    with tf.variable_scope('bbox') as scope:
#         with slim.arg_scope(bn_arg_scope()):
#             with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_trainingPH):  
                with slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(0.0005),
                                              biases_initializer=tf.constant_initializer(0.1),
                                              activation_fn=tf.nn.relu):    
                    [bz,hz,wz,cz]=featMap.get_shape().as_list()
                    net=slim.fully_connected(slim.flatten(featMap), 256, scope='attentionlayer1')
                    net=slim.fully_connected(net, 128, scope='attentionlayer2')
                    bbox=slim.fully_connected(net,4,activation_fn=None,scope='bboxl')
#                     net=slim.conv2d(featMap, 128, [3, 3], scope='conv0')
#                     net=slim.conv2d(net, 64, [3, 3],[2,2], scope='conv1')
#                     net=slim.conv2d(net, 32, [3, 3],[1,1], scope='conv2')
#                     net=slim.conv2d(net, 16, [3, 3],[2,2], scope='conv3')
#                     net=slim.conv2d(net, 8, [3, 3], scope='conv4')
#                     net=slim.flatten(net)
#                     net=slim.fully_connected(net, 4*hz*wz, scope='attentionlayer1')
#                     net=slim.fully_connected(net, 2*hz*wz, scope='attentionlayer2')
#                     net=slim.fully_connected(net, hz*wz, scope='attentionlayer3')
#                     bbox=slim.fully_connected(net,4,activation_fn=None,scope='bboxl')
                    return bbox

def CreateNetwork(imgPH,is_trainingPH):
    feat=build_encoder(imgPH,is_trainingPH)
    im_gen=build_decoder(feat,is_trainingPH)
    bbox_gen=predictBBox(feat,is_trainingPH)
    return im_gen,bbox_gen

def lossOp(im_gen,bbox_gen,imgPH,bboxPH):
    loss = 0.0*tf.reduce_mean(tf.square(im_gen - imgPH))
    print (bboxPH.get_shape().as_list())
    print (bbox_gen.get_shape().as_list())
    loss+=tf.reduce_mean(tf.square(bbox_gen - slim.flatten(bboxPH)))
    return loss

def train():    
    ##########################################################################################
    ##########################################################################################
    lr=0.01
    batch_size=128
    n_epochs= 200
    examples_to_show=10
    display_step=10
    #########################################################################################
    bboxes,indexes=chooseN_BBox(bbox,1)
    selectedImages=images[indexes[:]]
    total_examples= bboxes.shape[0]
    train_examples= int(total_examples*0.8)
    val_examples = total_examples - train_examples 
    train_iters_per_epoch = int((train_examples +batch_size -1)/ batch_size)
    val_iters_per_epoch = int((val_examples +batch_size -1) / batch_size)
    #########################################################################################
    imgPH= tf.placeholder(tf.float32,  shape=[batch_size,128,128,1])
    bboxPH= tf.placeholder(tf.float32,  shape=[batch_size,1,4])
    isTrainingPH= tf.placeholder(tf.bool,  shape=[])
    global_step = tf.Variable(0, trainable=False)

    decay_steps = int(train_iters_per_epoch * 1)
    learning_rate = tf.train.exponential_decay(lr,
                                  global_step,
                                  decay_steps,
                                  0.9,
                                  staircase=False)
    im_gen,bbox_gen=CreateNetwork(imgPH,isTrainingPH)
    loss=lossOp(im_gen,bbox_gen,imgPH,bboxPH)
    optim = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # optim = tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(loss,global_step=global_step)
    # optim = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        update_op = tf.group(*update_ops)
        # losses=tf.add_n(loss)
        loss = control_flow_ops.with_dependencies(update_ops, loss)

    train_image_idxs=np.array(range(train_examples))
    #########################################################################################
    #########################################################################################
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    #########################################################################################
    #########################################################################################

    rand_val_idxs = np.random.permutation(val_examples)+train_examples
    for e in range(n_epochs):
        rand_idxs = np.random.permutation(train_examples)
        curr_loss = 0
        for i in range(train_iters_per_epoch):
            if (i+1)*batch_size<len(rand_idxs):
                idxs=rand_idxs[i*batch_size:(i+1)*batch_size]
            else:
                idxs=rand_idxs[-batch_size:]
            input_bboxes=bboxes[idxs]
            inputImages=selectedImages[idxs].reshape([batch_size,128,128,1])
            feed_dict = {imgPH: inputImages,
                         bboxPH: input_bboxes,
                         isTrainingPH: True}
            _, l, bboxPredict,lr_n = sess.run([optim, loss, bbox_gen,learning_rate], feed_dict)
            curr_loss+=l
            if i%display_step==0:
                print (bboxPredict[0],input_bboxes[0])
        avgLoss=curr_loss/train_iters_per_epoch
        print("Epoch:",'%04d'% (e+1),
              "learning rate","{:.9f}".format(lr_n),
              "loss=", "{:.9f}".format(avgLoss))
        sys.stdout.flush()
        ##################################################################
        ######################Validation##################################
        idxs_val=rand_val_idxs[0:batch_size]
        input_bboxes_val=bboxes[idxs_val]
        inputImages_val=selectedImages[idxs_val].reshape([batch_size,128,128,1])
        feed_dict = {imgPH: inputImages_val,
                     bboxPH: input_bboxes_val,
                     isTrainingPH: False}
        l, bboxPredict = sess.run([loss, bbox_gen], feed_dict)
        for i in range(10):
            print (bboxPredict[i],input_bboxes_val[i],l)
        sys.stdout.flush()


if __name__ == "__main__":
    train()