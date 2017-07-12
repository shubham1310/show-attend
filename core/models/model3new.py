# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim


class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[464, 512], dim_embed=512, dim_hidden=1024, n_time_step=16, 
                  prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True, imshape=128, channels = 1):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM. 
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """
        
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, imshape, imshape, channels])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
    
    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)  
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)  
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha
  
    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context') 
            return context, beta
  
    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M*3, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                # w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                #  += tf.matmul(, w_ctx2out)
                h_logits = tf.concat( [h_logits, context],1 )

            h_logits = tf.concat( [h_logits, x ],1 )
            # if self.prev2out:
            #     h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits


    def build_model(self):

        features = self.features
        captions = self.captions
        batch_size = tf.shape(features)[0]

        captions_in = captions[:, :self.T]      
        captions_out = captions[:, 1:]  
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        with tf.variable_scope('model') as scope:   
            x_image = tf.reshape(features, [-1,128,128,1])
            h_conv1 = tf.nn.relu(slim.conv2d(x_image, 32, [3, 3], scope='conv1'))
            h_conv2 = tf.nn.relu(slim.conv2d(h_conv1, 64, [3, 3], scope='conv2'))
            h_pool1 = slim.max_pool2d(h_conv2, [2, 2], scope='pool1')

            h_conv3 = tf.nn.relu(slim.conv2d(h_pool1, 128, [3, 3], scope='conv3'))
            h_conv4 = tf.nn.relu(slim.conv2d(h_conv3, 128, [2, 2], scope='conv4'))
            h_pool2 = slim.max_pool2d(h_conv4, [2, 2], scope='pool2')
            
            h_conv5 = tf.nn.relu(slim.conv2d(h_pool2, 256, [3, 3], scope='conv5'))
            h_conv6 = tf.nn.relu(slim.conv2d(h_conv5, 512, [3, 3], scope='conv6'))
            h_pool3 = slim.max_pool2d(h_conv6, [2, 2], scope='pool3')

        resh = tf.reshape(h_pool3,[-1,h_pool3.get_shape().as_list()[1]**2,512])

        with tf.variable_scope('decoder') as scope:
                dectran1 = slim.conv2d_transpose(h_pool3, 512, [3,3], [2,2] , scope = 'deconvtran1') #print dectran1.get_shape()
                deconv1 = tf.nn.relu(slim.conv2d(dectran1, 256, [3, 3], scope='deconv1')) #print deconv1.get_shape()
                deconv2 = tf.nn.relu(slim.conv2d(deconv1, 128, [3, 3], scope='deconv2')) #print deconv2.get_shape()

                dectran2 = slim.conv2d_transpose(deconv2, 128, [3,3], [2,2] , scope = 'deconvtran2') #print dectran2.get_shape()
                deconv3 = tf.nn.relu(slim.conv2d(dectran2, 128, [3, 3], scope='deconv3')) #print deconv3.get_shape()
                deconv4 = tf.nn.relu(slim.conv2d(deconv3, 64, [3, 3], scope='deconv4')) #print deconv4.get_shape()

                dectran3 = slim.conv2d_transpose(deconv4, 64, [3,3], [2,2] , scope = 'deconvtran3') #print dectran3.get_shape()
                deconv5 = tf.nn.relu(slim.conv2d(dectran3, 32, [3, 3], scope='deconv5')) #print deconv5.get_shape()
                deconv6 = tf.nn.relu(slim.conv2d(deconv5, 1, [3, 3], scope='deconv6')) #print deconv6.get_shape()

        resh1 = tf.reshape(x_image,[-1,deconv6.get_shape().as_list()[1]**2])
        resh2 = tf.reshape(deconv6,[-1,deconv6.get_shape().as_list()[1]**2])

        features = tf.contrib.layers.batch_norm(inputs=resh, 
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=('train'=='train'),
                                            updates_collections=None,
                                            scope=('conv_features'+'batch_norm'))
        
        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions_in)
        features_proj = self._project_features(features=features)

        loss = 0.1 *tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(resh1, resh2))))
        alpha_list = []
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0)) 

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat( [x[:,t,:], context],1 ), state=[c, h])

            logits = self._decode_lstm(x[:,t,:], h, context, dropout=self.dropout, reuse=(t!=0))
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=captions_out[:, t]) * mask[:, t])
           
        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((17./self.L - alphas_all) ** 2)     
            loss += alpha_reg

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        features = self.features
        
        # batch normalize feature vectors
        with tf.variable_scope('model') as scope:
            # scope.reuse_variables()
            x_image = tf.reshape(features, [-1,128,128,1])
            h_conv1 = tf.nn.relu(slim.conv2d(x_image, 32, [3, 3], scope='conv1'))
            h_conv2 = tf.nn.relu(slim.conv2d(h_conv1, 64, [3, 3], scope='conv2'))
            h_pool1 = slim.max_pool2d(h_conv2, [2, 2], scope='pool1')


            h_conv3 = tf.nn.relu(slim.conv2d(h_pool1, 128, [3, 3], scope='conv3'))
            h_conv4 = tf.nn.relu(slim.conv2d(h_conv3, 128, [2, 2], scope='conv4'))
            h_pool2 = slim.max_pool2d(h_conv4, [2, 2], scope='pool2')
            

            h_conv5 = tf.nn.relu(slim.conv2d(h_pool2, 256, [3, 3], scope='conv5'))
            h_conv6 = tf.nn.relu(slim.conv2d(h_conv5, 512, [3, 3], scope='conv6'))
            h_pool3 = slim.max_pool2d(h_conv6, [2, 2], scope='pool3')
            # print h_pool3.get_shape()
            # print h_conv6.get_shape()
            # print h_conv5.get_shape()
            # print h_pool2.get_shape()
            # print h_conv4.get_shape()
            # print h_conv3.get_shape()
            # print h_pool1.get_shape()
            # print h_conv2.get_shape()
            # print h_conv1.get_shape()
            # print x_image.get_shape()

        resh = tf.reshape(h_pool3,[-1,h_pool3.get_shape().as_list()[1]**2,512])

        # while 1:
        #     pass

        features = tf.contrib.layers.batch_norm(inputs=resh, 
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=('test'=='train'),
                                            updates_collections=None,
                                            scope=('conv_features'+'batch_norm'))
        
        
        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H,reuse = True)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)  
          
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0)) 
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat( [x, context],1), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=(t!=0))
            sampled_word = tf.argmax(logits, 1)       
            sampled_word_list.append(sampled_word)     

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return alphas, betas, sampled_captions
