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
import numpy as np
from tensorflow.python.ops import array_ops
########################Added For Intersection RNN############################
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
####################################################################################

_checked_scope = core_rnn_cell_impl._checked_scope  # pylint: disable=protected-access
_linear = core_rnn_cell_impl._linear
class IntersectionRNNCell(core_rnn_cell.RNNCell):
  """Intersection Recurrent Neural Network (+RNN) cell.

  Architecture with coupled recurrent gate as well as coupled depth
  gate, designed to improve information flow through stacked RNNs. As the
  architecture uses depth gating, the dimensionality of the depth
  output (y) also should not change through depth (input size == output size).
  To achieve this, the first layer of a stacked Intersection RNN projects
  the inputs to N (num units) dimensions. Therefore when initializing an
  IntersectionRNNCell, one should set `num_in_proj = N` for the first layer
  and use default settings for subsequent layers.

  This implements the recurrent cell from the paper:

    https://arxiv.org/abs/1611.09913

  Jasmine Collins, Jascha Sohl-Dickstein, and David Sussillo.
  "Capacity and Trainability in Recurrent Neural Networks" Proc. ICLR 2017.

  The Intersection RNN is built for use in deeply stacked
  RNNs so it may not achieve best performance with depth 1.
  """

  def __init__(self, num_units, num_in_proj=None,
               initializer=None, forget_bias=1.0,
               y_activation=nn_ops.relu, reuse=None):
    """Initialize the parameters for an +RNN cell.

    Args:
      num_units: int, The number of units in the +RNN cell
      num_in_proj: (optional) int, The input dimensionality for the RNN.
        If creating the first layer of an +RNN, this should be set to
        `num_units`. Otherwise, this should be set to `None` (default).
        If `None`, dimensionality of `inputs` should be equal to `num_units`,
        otherwise ValueError is thrown.
      initializer: (optional) The initializer to use for the weight matrices.
      forget_bias: (optional) float, default 1.0, The initial bias of the
        forget gates, used to reduce the scale of forgetting at the beginning
        of the training.
      y_activation: (optional) Activation function of the states passed
        through depth. Default is 'tf.nn.relu`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    self._num_units = num_units
    self._initializer = initializer
    self._forget_bias = forget_bias
    self._num_input_proj = num_in_proj
    self._y_activation = y_activation
    self._reuse = reuse

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Run one step of the Intersection RNN.

    Args:
      inputs: input Tensor, 2D, batch x input size.
      state: state Tensor, 2D, batch x num units.
      scope: VariableScope for the created subgraph; defaults to
        "intersection_rnn_cell"

    Returns:
      new_y: batch x num units, Tensor representing the output of the +RNN
        after reading `inputs` when previous state was `state`.
      new_state: batch x num units, Tensor representing the state of the +RNN
        after reading `inputs` when previous state was `state`.

    Raises:
      ValueError: If input size cannot be inferred from `inputs` via
        static shape inference.
      ValueError: If input size != output size (these must be equal when
        using the Intersection RNN).
    """
    sigmoid = math_ops.sigmoid
    tanh = math_ops.tanh

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    with _checked_scope(self, scope or "intersection_rnn_cell",
                        initializer=self._initializer, reuse=self._reuse):
      # read-in projections (should be used for first layer in deep +RNN
      # to transform size of inputs from I --> N)
      if input_size.value != self._num_units:
        if self._num_input_proj:
          with vs.variable_scope("in_projection"):
            inputs = _linear(inputs, self._num_units, True)
        else:
          raise ValueError("Must have input size == output size for "
                           "Intersection RNN. To fix, num_in_proj should "
                           "be set to num_units at cell init.")

      n_dim = i_dim = self._num_units
      cell_inputs = array_ops.concat([inputs, state], 1)
      rnn_matrix = _linear(cell_inputs, 2*n_dim + 2*i_dim, True)

      gh_act = rnn_matrix[:, :n_dim]                           # b x n
      h_act = rnn_matrix[:, n_dim:2*n_dim]                     # b x n
      gy_act = rnn_matrix[:, 2*n_dim:2*n_dim+i_dim]            # b x i
      y_act = rnn_matrix[:, 2*n_dim+i_dim:2*n_dim+2*i_dim]     # b x i

      h = tanh(h_act)
      y = self._y_activation(y_act)
      gh = sigmoid(gh_act + self._forget_bias)
      gy = sigmoid(gy_act + self._forget_bias)

      new_state = gh * state + (1.0 - gh) * h  # passed thru time
      new_y = gy * inputs + (1.0 - gy) * y  # passed thru depth

    return new_y, new_state


def BidirectionalVerticalRnn(net,statesize,scope):
    fwdGRUCell=IntersectionRNNCell(statesize,statesize,reuse=tf.get_variable_scope().reuse)
    bwdGRUCell=IntersectionRNNCell(statesize,statesize,reuse=tf.get_variable_scope().reuse)
    [bsz,hsz,wsz,csz]=net.get_shape().as_list()
    map_bxYC=tf.reshape(tf.transpose(net,[0,2,1,3]),[-1,hsz,csz])
    inputs = tf.unstack(map_bxYC,axis=1)
    output,fws,bws = tf.contrib.rnn.static_bidirectional_rnn(fwdGRUCell,bwdGRUCell,inputs,dtype=tf.float32,scope=scope)
    map_bxYC=tf.stack(output,axis=1);
    map_bxYC=tf.reshape(map_bxYC,[bsz,wsz,hsz,-1])
    net=tf.transpose(map_bxYC,[0,2,1,3])
    return net

def BidirectionalHorzRnn(net,statesize,scope):
    fwdGRUCell=IntersectionRNNCell(statesize,statesize,reuse=tf.get_variable_scope().reuse)
    bwdGRUCell=IntersectionRNNCell(statesize,statesize,reuse=tf.get_variable_scope().reuse)
    [bsz,hsz,wsz,csz]=net.get_shape().as_list()
    map_byXC=tf.reshape(net,[-1,wsz,csz])
    inputs = tf.unstack(map_byXC,axis=1)
    output,fws,bws = tf.contrib.rnn.static_bidirectional_rnn(fwdGRUCell,bwdGRUCell,inputs,dtype=tf.float32,scope=scope)
    map_byXC=tf.stack(output,axis=1);
    net=tf.reshape(map_byXC,[bsz,hsz,wsz,-1])
    return net

def mapRnnLayer(net,statesize,scope):
    net=BidirectionalVerticalRnn(net,statesize,scope=scope+'_VRNN')
    net=BidirectionalHorzRnn(net,statesize,scope=scope+'_HRNN')
    return net




class CaptionGenerator(object):
    def __init__(self, dim_feature=[464, 512], dim_embed=512, dim_hidden=1024, n_time_step=16, 
                  prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, flag =1, batch_size=64, dropout=True, imshape=128, channels = 1):
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
      

        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.flag=flag
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = 128
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = np.array([0.0,0.0,0.0,0.0])
        self._null = np.array([-1.0,-1.0,-1.0,-1.0])
        self.batch_size =batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and bbox
        self.features = tf.placeholder(tf.float32, [batch_size, imshape, imshape, channels])
        self.bbox = tf.placeholder(tf.int32, [batch_size, self.T + 1, 4])
    
    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            # w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            # b_h = tf.get_variable('b_h', [], initializer=self.const_initializer)
            h = tf.nn.tanh(slim.fully_connected(features_mean, self.H, scope = 'lstmfc1',activation_fn=None))
            # h = tf.nn.tanh(tf.matmul(, w_h) + b_h)

            # w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            # b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            # c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            c = tf.nn.tanh(slim.fully_connected(features_mean, self.H, scope = 'lstmfc2',activation_fn=None))
            return c, h

    # def _word_embedding(self, inputs, reuse=False):
    #     with tf.variable_scope('word_embedding', reuse=reuse):
    #         w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
    #         x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
    #         return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            # w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = slim.fully_connected(features_flat, self.D, normalizer_fn=lambda x: x, scope = 'features_proj',activation_fn=None)
            # features_proj = tf.matmul(features_flat, w)  
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            # w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            # w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(slim.fully_connected(h, self.D, normalizer_fn=lambda x: x,activation_fn=None, scope = 'attentionlayer'), 1) + b)    # (N, L, D)
                
            out_att = tf.reshape(slim.fully_connected(tf.reshape(h_att, [-1, self.D]), 1, normalizer_fn=lambda x: x,activation_fn=None, scope = 'attentionlayer2'), [-1, self.L])
            # out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)  

            # print features.get_shape(),features.get_shape()[2]
            context = tf.reshape( tf.expand_dims(alpha, 2), [-1,16,16,1])

            context = tf.nn.relu(slim.conv2d(context, 32, [3, 3], scope='attentionconv1'))
            context = tf.reshape(context,[-1,16*16*32])
            context = slim.fully_connected(context, int(features.get_shape()[2]),activation_fn=None, scope = 'attentionlayerfc')
            # context =  tf.nn.relu(slim.conv2d(context, 1, [3, 3], scope='attentionconv2'))

            
            # print alpha.get_shape()
            # context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            # print context.get_shape()
            return context, alpha
  
    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            # w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            # b = tf.get_variable('b', [1], initializer=self.const_initializer)


            beta = tf.nn.sigmoid(slim.fully_connected(h, 1, scope = 'selectorfc',activation_fn=None), 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context') 
            return context, beta
  
    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            # w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            # b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            # w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            # b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)
            # w_last = tf.get_variable('w_last', [self.V, 4], initializer=self.weight_initializer)
            # b_last = tf.get_variable('b_last', [4], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)

            h_logits= slim.fully_connected(h, self.M, scope = 'decodefc',activation_fn=None)
            # h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                # w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += slim.fully_connected(context, self.M,  normalizer_fn=lambda x: x, scope = 'decodefc2',activation_fn=None)
                # h_logits += tf.matmul(, w_ctx2out)

            if self.prev2out:
                h_logits += x
            # h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = slim.fully_connected(h_logits, self.V, scope = 'decodefcOut',activation_fn=None)# tf.matmul(, w_out) + b_out

            last_logits = slim.fully_connected(out_logits, 4, scope = 'decodefcLast',activation_fn=None)# tf.matmul(, w_last) + b_last
            return last_logits


    def build_model(self):

        features = self.features
        bbox = self.bbox
        batch_size = tf.shape(features)[0]

        bbox_in = tf.to_float(bbox[:, :self.T])      
        bbox_out = tf.to_float(bbox[:, 1:]) 
        # print bbox_out.shape, bbox.shape
        mask = tf.to_float(tf.not_equal(bbox_out, self._null))

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

                dectran2 = slim.conv2d_transpose(deconv2, 128, [3,3], [2,2] , scope = 'deconvtran2')#print dectran2.get_shape()
                deconv3 = tf.nn.relu(slim.conv2d(dectran2, 128, [3, 3], scope='deconv3'))#print deconv3.get_shape()
                deconv4 = tf.nn.relu(slim.conv2d(deconv3, 64, [3, 3], scope='deconv4'))# print deconv4.get_shape()

                dectran3 = slim.conv2d_transpose(deconv4, 64, [3,3], [2,2] , scope = 'deconvtran3')#print dectran3.get_shape()
                deconv5 = tf.nn.relu(slim.conv2d(dectran3, 32, [3, 3], scope='deconv5'))#print deconv5.get_shape()
                deconv6 = tf.nn.relu(slim.conv2d(deconv5, 1, [3, 3], scope='deconv6'))#print deconv6.get_shape()

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
        x = tf.to_float(bbox_in)

        x= slim.fully_connected(x,512, scope = 'fc')

        features_proj = self._project_features(features=features)

        loss = 0.01 *tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(resh1, resh2))))
        # alpha_list = []
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)

        # for t in range(self.T):
        #     context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
        #     alpha_list.append(alpha)

        #     if self.selector:
        #         context, beta = self._selector(context, h, reuse=(t!=0)) 

        #     # print context.get_shape()
        #     with tf.variable_scope('lstm', reuse=(t!=0)):
        #         _, (c, h) = lstm_cell(inputs=tf.concat( [x[:,t,:], context],1 ), state=[c, h])

        #     logits = self._decode_lstm(x[:,t,:], h, context, dropout=self.dropout, reuse=(t!=0))

        net = mapRnnLayer(h_pool3, 32,scope='net')
        # print net.get_shape()
        pred = tf.nn.relu(slim.conv2d(net, 4, [3, 3], scope='predconv'))
        # print pred.get_shape()
        # pred = tf.slice(pred, [ 0, 15, 15 , 0] , [-1, 1, 1, 4] )

        # pred = tf.reshape(pred,[-1,4])
        # if self.flag == 0:
        #     print "Fully connected layer activated"
        #     pred = slim.fully_connected(pred, 4, scope = 'fcpred',activation_fn=None)
        # pred = pred[:, pred.get_shape().as_list()[1]-1, pred.get_shape().as_list()[1]-1, 4]
        loss += tf.reduce_sum(tf.reduce_mean(tf.square(tf.subtract(pred, bbox_out[:, ])* mask[:, ])))
           
        # if self.alpha_c > 0:
        #     alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        #     alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
        #     alpha_reg = self.alpha_c * tf.reduce_sum((16./self.L - alphas_all) ** 2)     
        #     loss += alpha_reg

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        features = self.features
        # max_len = self.n_time_step +1
        # batch normalize feature vectors

        with tf.variable_scope('model') as scope:
            x_image = tf.reshape(features, [-1,128,128,1])      # print x_image.get_shape()
            h_conv1 = tf.nn.relu(slim.conv2d(x_image, 32, [3, 3], scope='conv1'))   # print h_conv1.get_shape()
            h_conv2 = tf.nn.relu(slim.conv2d(h_conv1, 64, [3, 3], scope='conv2'))   # print h_conv2.get_shape()   
            h_pool1 = slim.max_pool2d(h_conv2, [2, 2], scope='pool1')   # print h_pool1.get_shape()


            h_conv3 = tf.nn.relu(slim.conv2d(h_pool1, 128, [3, 3], scope='conv3'))  # print h_conv3.get_shape()
            h_conv4 = tf.nn.relu(slim.conv2d(h_conv3, 128, [2, 2], scope='conv4'))  # print h_conv4.get_shape()
            h_pool2 = slim.max_pool2d(h_conv4, [2, 2], scope='pool2')   # print h_pool2.get_shape()
            

            h_conv5 = tf.nn.relu(slim.conv2d(h_pool2, 256, [3, 3], scope='conv5')) # print h_conv5.get_shape()
            h_conv6 = tf.nn.relu(slim.conv2d(h_conv5, 512, [3, 3], scope='conv6')) # print h_conv6.get_shape()
            h_pool3 = slim.max_pool2d(h_conv6, [2, 2], scope='pool3')   # print h_pool3.get_shape()
                      

        resh = tf.reshape(h_pool3,[-1,h_pool3.get_shape().as_list()[1]**2,512])


        features = tf.contrib.layers.batch_norm(inputs=resh, 
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=('test'=='train'),
                                            updates_collections=None,
                                            scope=('conv_features'+'batch_norm'))
        
        net = mapRnnLayer(h_pool3, 32,scope='net')
        # print net.get_shape()
        pred = tf.nn.relu(slim.conv2d(net, 4, [3, 3], scope='predconv'))
        return pred
        # print pred.get_shape()
        # pred = tf.slice(pred, [ 0, 15, 15 , 0] , [-1, 1, 1, 4] ) 
        # pred = tf.reshape(pred,[-1,4])
        # if self.flag == 0:
        #     print "Fully connected layer activated"
        #     pred = slim.fully_connected(pred, 4, scope = 'fcpred',activation_fn=None)
        # pred = pred[:, pred.get_shape().as_list()[1]-1, pred.get_shape().as_list()[1]-1, 4]
        # loss += tf.reduce_sum(tf.reduce_mean(tf.square(tf.subtract(pred, bbox_out[:, ])* mask[:, ])))


        # c, h = self._get_initial_lstm(features=features)
        # features_proj = self._project_features(features=features)

        # sampled_word_list = []
        # alpha_list = []
        # beta_list = []
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H,reuse = True)

        # for t in range(max_len):
        #     if t == 0:
        #         x = tf.fill([tf.shape(features)[0], 4], tf.to_float(0))
        #     else:
        #         x = sampled_word

        #     x=tf.to_float(x)
        #     x= slim.fully_connected(x,512, scope = 'fc')
        #     context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
        #     alpha_list.append(alpha)

        #     if self.selector:
        #         context, beta = self._selector(context, h, reuse=(t!=0)) 
        #         beta_list.append(beta)

        #     with tf.variable_scope('lstm', reuse=(t!=0)):
        #         _, (c, h) = lstm_cell(inputs=tf.concat( [x, context],1), state=[c, h])

        #     logits = self._decode_lstm(x, h, context, reuse=(t!=0))
        #     sampled_word = logits       
        #     sampled_word_list.append(sampled_word)     

        # alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        # betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        # # generated_boxes = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        # # print "here"
        # # print alphas.get_shape() #
        # generated_boxes = tf.transpose(sampled_word_list, (1, 0, 2))     # (N, max_len)
        # return tf.reshape(alphas, [-1, max_len, 16,16]) , betas, generated_boxes
        

