import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os, random
import cPickle as pickle
from scipy import ndimage
# from utils import *
# from bleu import evaluate
from PIL import Image
import numpy as np


class CaptioningSolver(object):
    def __init__(self, model, path, bbox,  **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path 
            - model_path: String; model path for saving 
            - test_model: String; model path for test 
        """

        self.model = model
        self.n_time_step = kwargs.pop('n_time_step', 1) 
        self.path=path
        self.bbox=bbox
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 50)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer
        else:
            self.optimizer = tf.train.GradientDescentOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def train(self, chunk=0):
        # train/val dataset


        features = np.load(self.path + 'trainRandomTRs1.npy', mmap_mode='r')
        n= self.bbox.shape[0]
        n_examples= int(n*0.8)
        val_examples = n - n_examples 
        
        # n_examples = self.captions.shape[0]
        n_iters_per_epoch = int(n_examples / self.batch_size)


        bbox = self.bbox
        image_idxs = np.array([i for i in range(n_examples)])

        n_iters_val = int(val_examples / self.batch_size)

        # build graphs for training model and sampling captions
        loss = self.model.build_model()
        # print (loss)
        # print len(features), len(captions)
        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
            
        tf.get_variable_scope().reuse_variables()
        generated_boxes = self.model.build_sampler(max_len=self.n_time_step)

        # summary op
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        # for grad, var in grads_and_vars:
        #     tf.summary.histogram(var.op.name + '/gradient', grad)

        summary_op = tf.summary.merge_all()

        print "The number of epoch: %d" % self.n_epochs
        print "Data size: %d" % n_examples
        print "Batch size: %d" % self.batch_size
        print "Iterations per epoch: %d" % n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        init = tf.initialize_all_variables()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(
                self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                bboxs = bbox[rand_idxs]
                image_idxs = image_idxs[rand_idxs]

                # print "epoch number: %d " %e
                for i in range(n_iters_per_epoch):
                    # print "Iteration number %d" % i
                    bboxs_batch = bboxs[i *self.batch_size:(i + 1) * self.batch_size]
                    image_idxs_batch = image_idxs[i * self.batch_size : (i + 1) * self.batch_size]
                    features_batch = features[image_idxs_batch]
                    # bboxs_batch = bboxs[image_idxs_batch]
                    # print len(captions_batch), len(features_batch)
                    features_batchn = []
                    for k in range(len(features_batch)):
                        features_batchn.append(np.expand_dims(features_batch[k],3))

                    # features_batchn = np.array(features_batchn)
                    feed_dict = {self.model.features: features_batchn,
                                 self.model.bbox: bboxs_batch}
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(
                            summary, (e + chunk * 10) * n_iters_per_epoch + i)

                    if (i + 1) % self.print_every == 0:
                        print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" % (e + 1 + chunk * 10, i + 1, l)
                        ground_truths = bboxs[image_idxs ==image_idxs_batch[0]]
                        # decoded = decode_captions(
                        #     ground_truths, self.model.idx_to_word)
                        print "Ground:\n",
                        fl=1
                        # print ground_truths[0][0]
                        for gt in ground_truths[0]:
                            if fl==1:
                                fl=0
                            else:
                                print gt
                        print '\n',
                        gen_bboxs = sess.run(generated_boxes, feed_dict)
                        # decoded = decode_captions(
                        #     gen_caps, self.model.idx_to_word)
                        print "Generated:\n",
                        k = gen_bboxs[0]
                        print "[%0.2f, %.2f, %.2f, %.2f]" %(k[0],k[1],k[2],k[3])
                        print "\n"

                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0

                # print out BLEU scores and file write
                if self.print_bleu and (e % 5 ==0):
                    all_gen_box = np.ndarray((n_iters_val*self.batch_size,4))
                    # alp=[]
                    for i in range(n_iters_val):
                        features_batch = features[n_examples + i *self.batch_size: n_examples + (i + 1) * self.batch_size]
                        features_batchn = []
                        for k in range(len(features_batch)):
                            features_batchn.append(np.expand_dims(features_batch[k],3))

                        # features_batchn = np.array(features_batchn)
                        feed_dict = {self.model.features: features_batchn}

                        gen_box = sess.run(
                             generated_boxes,  feed_dict=feed_dict)
                        all_gen_box[i *self.batch_size:(i + 1) * self.batch_size] = gen_box
                    # print all_gen_box[0]
                        # for b in alpha:
                        #     alp.append(b)

                    # for i in all_gen_box:
                    #     print i
                    # all_decoded = decode_captions(
                    #     all_gen_box, self.model.idx_to_word)

                    for i in range(3):
                        j = random.randint(0,n_iters_val*self.batch_size)
                        while j >= len(all_gen_box):
                            j=j/10
                        print "\nGround:\n",
                        # print self.bbox[n_examples + j]
                        k =self.bbox[n_examples + j][1]
                        print "[%0.2f, %.2f, %.2f, %.2f]" %(k[0],k[1],k[2],k[3])
                        print '\n',
                        print "Generated:\n",
                        k=all_gen_box[j] 
                        print "[%0.2f, %.2f, %.2f, %.2f]" %(k[0],k[1],k[2],k[3])
                        # print '\n'
                        # for k in range(6):
                        #     img = (((alp[j][k] - alp[j][k].min()) / (alp[j][k].max() - alp[j][k].min())) * 255.9).astype(np.uint8)
                        #     img=Image.fromarray(alp[j][k], 'L')
                        #     if not os.path.exists('imgs/alp' + str(e) +'/' + str(j)):
                        #         os.makedirs('imgs/alp' + str(e) +'/' + str(j))
                        #     img.save('imgs/alp' + str(e) +'/' + str(j) +'/' + str(k) +'.jpg')

                        # plt.plot(alp[j]); 
                        # plt.savefig(  )
                        # print 
                       

                # save model's parameters
                if (e + 1) % self.save_every == 0:
                    saver.save(sess, os.path.join(
                        self.model_path, 'model'), global_step=e + 1 + chunk * 10)
                    print "model-%s saved." % (e + 1 + chunk * 10)

    def test(self, data, split='test', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17) 
            - image_idxs: Indices for mapping caption to image of shape (24210, ) 
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(
            max_len=17)    # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch, image_files = sample_coco_minibatch(
                data, self.batch_size)
            feed_dict = {self.model.features: features_batch}
            # (N, max_len, L), (N, max_len)
            alps, bts, sam_cap = sess.run(
                [alphas, betas, sampled_captions], feed_dict)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)

            if attention_visualization:
                for n in range(10):
                    print "Sampled Caption: %s" % decoded[n]

                    # Plot original image
                    img = ndimage.imread(image_files[n])
                    plt.subplot(4, 5, 1)
                    plt.imshow(img)
                    plt.axis('off')

                    # Plot images with attention weights
                    words = decoded[n].split(" ")
                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(4, 5, t + 2)
                        plt.text(0, 1, '%s(%.2f)' % (
                            words[t], bts[n, t]), color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(img)
                        alp_curr = alps[n, t, :].reshape(14, 14)
                        alp_img = skimage.transform.pyramid_expand(
                            alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        plt.axis('off')
                    plt.show()

            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], 20))
                num_iter = int(
                    np.ceil(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i *
                                              self.batch_size:(i + 1) * self.batch_size]
                    feed_dict = {self.model.features: features_batch}
                    all_sam_cap[i * self.batch_size:(i + 1) * self.batch_size] = sess.run(
                        sampled_captions, feed_dict)
                all_decoded = decode_captions(
                    all_sam_cap, self.model.idx_to_word)
                save_pickle(
                    all_decoded, "./data/%s/%s.candidate.captions.pkl" % (split, split))
