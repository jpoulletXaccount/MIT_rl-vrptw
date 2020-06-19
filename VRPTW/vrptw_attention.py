
import tensorflow as tf

class AttentionVRPTWActor(object):
    """A generic attention module for the attention in vrptw model"""
    def __init__(self, dim, use_tanh=False, C=10,_name='Attention',_scope=''):
        self.use_tanh = use_tanh
        self._scope = _scope

        with tf.variable_scope(_scope+_name):
            # self.v: is a variable with shape [1 x dim]
            self.v = tf.get_variable('v',[1,dim],
                       initializer=tf.contrib.layers.xavier_initializer())
            self.v = tf.expand_dims(self.v,2)   # Todo 2 may be weird

        self.emb_d = tf.layers.Conv1D(dim,1,_scope=_scope+_name+'/emb_d' ) #conv1d of kernel size = dim, stride = 1
        self.emb_ld = tf.layers.Conv1D(dim,1,_scope=_scope+_name+'/emb_ld' ) #conv1d_2
        self.emb_time = tf.layers.Conv1D(dim,1,_scope=_scope+_name+'/emb_time' ) #conv1d_ for time

        self.project_d = tf.layers.Conv1D(dim,1,_scope=_scope+_name+'/proj_d' ) #conv1d_1
        self.project_t = tf.layers.Conv1D(dim,1,_scope=_scope+_name+'/proj_t' ) #conv1d_1
        self.project_ld = tf.layers.Conv1D(dim,1,_scope=_scope+_name+'/proj_ld' ) #conv1d_3
        self.project_query = tf.layers.Dense(dim,_scope=_scope+_name+'/proj_q' ) # fully connected layer, activation is linear
        self.project_ref = tf.layers.Conv1D(dim,1,_scope=_scope+_name+'/proj_ref' ) #conv1d_4


        self.C = C  # tanh exploration parameter
        self.tanh = tf.nn.tanh      # activation function hyperbolique tangente (output in ]-1,1[

    def __call__(self, query, ref, env):
        """
        This function gets a query tensor and ref tensor and returns the logit op.
        Args:
            query: is the hidden state of the decoder at the current
                time step. [batch_size x dim]
            ref: the set of hidden states from the encoder.
                [batch_size x max_time x dim]

        Returns:
            e: convolved ref with shape [batch_size x max_time x dim]
            logits: [batch_size x max_time]
        """
        # get the current demand, time and load values from environment
        demand = env.demand
        load = env.load
        current_time = env.time
        max_time = tf.shape(demand)[1]  # number nodes


        # embed demand and project it
        # emb_d:[batch_size x max_time x dim ]
        emb_d = self.emb_d(tf.expand_dims(demand,2))
        # d:[batch_size x max_time x dim ]
        d = self.project_d(emb_d)

        # embed load - demand
        # emb_ld:[batch_size*beam_width x max_time x hidden_dim]
        emb_ld = self.emb_ld(tf.expand_dims(tf.tile(tf.expand_dims(load,1),[1,max_time])-
                                              demand,2))
        # ld:[batch_size*beam_width x hidden_dim x max_time ]
        ld = self.project_ld(emb_ld)

        # embed current time Todo may be worth doing the beginning tw  and end tw as well
        emb_time = self.emb_time(tf.expand_dims(tf.expand_dims(current_time,1),2))
        t = self.project_t(emb_time)

        # expanded_q,e: [batch_size x max_time x dim]
        e = self.project_ref(ref)
        q = self.project_query(query) #[batch_size x dim]
        expanded_q = tf.tile(tf.expand_dims(q,1),[1,max_time,1])

        # v_view:[batch_size x dim x 1]
        v_view = tf.tile( self.v, [tf.shape(e)[0],1,1])

        # u : [batch_size x max_time x dim] * [batch_size x dim x 1] =
        #       [batch_size x max_time]
        u = tf.squeeze(tf.matmul(self.tanh(expanded_q + e + d + ld + t), v_view),2)

        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u

        return e, logits


class AttentionVRPTWCritic(object):
    """A generic attention module for the attention in vrp model"""
    def __init__(self, dim, use_tanh=False, C=10,_name='Attention',_scope=''):

        self.use_tanh = use_tanh
        self._scope = _scope

        with tf.variable_scope(_scope+_name):
            # self.v: is a variable with shape [1 x dim]
            self.v = tf.get_variable('v',[1,dim],
                       initializer=tf.contrib.layers.xavier_initializer())
            self.v = tf.expand_dims(self.v,2)

        self.emb_d = tf.layers.Conv1D(dim,1,_scope=_scope+_name +'/emb_d') #conv1d
        self.emb_end_tw = tf.layers.Conv1D(dim,1,_scope=_scope+_name +'/emb_end_tw') #conv1d
        self.emb_begin_tw = tf.layers.Conv1D(dim,1,_scope=_scope+_name +'/emb_begin_tw') #conv1d
        self.project_d = tf.layers.Conv1D(dim,1,_scope=_scope+_name +'/proj_d') #conv1d_1
        self.project_end_tw = tf.layers.Conv1D(dim,1,_scope=_scope+_name +'/proj_end_tw') #conv1d_1
        self.project_begin_tw = tf.layers.Conv1D(dim,1,_scope=_scope+_name +'/proj_begin_tw') #conv1d_1

        self.project_query = tf.layers.Dense(dim,_scope=_scope+_name +'/proj_q') #
        self.project_ref = tf.layers.Conv1D(dim,1,_scope=_scope+_name +'/proj_e') #conv1d_2

        self.C = C  # tanh exploration parameter
        self.tanh = tf.nn.tanh

    def __call__(self, query, ref, env):
        """
        This function gets a query tensor and ref rensor and returns the logit op.
        Args:
            query: is the hidden state of the decoder at the current
                time step. [batch_size x dim]
            ref: the set of hidden states from the encoder.
                [batch_size x max_time x dim]

            env: keeps demand ond load values and help decoding. Also it includes mask.
                env.mask: a matrix used for masking the logits and glimpses. It is with shape
                         [batch_size x max_time]. Zeros in this matrix means not-masked nodes. Any
                         positive number in this mask means that the node cannot be selected as next
                         decision point.
                env.demands: a list of demands which changes over time.

        Returns:
            e: convolved ref with shape [batch_size x max_time x dim]
            logits: [batch_size x max_time]
        """
        # we need the first demand value for the critic
        # TOdo why end tw and beginning tw? or current time ??
        demand = env.input_data[:,:,-1]
        end_tw = env.input_data[:,:,3]
        begin_tw = env.input_data[:,:,2]
        max_time = tf.shape(demand)[1]

        # embed demand and project it
        # emb_d:[batch_size x max_time x dim ]
        emb_d = self.emb_d(tf.expand_dims(demand,2))
        # d:[batch_size x max_time x dim ]
        d = self.project_d(emb_d)

        # embed tw and project then
        emb_end_tw = self.emb_begin_tw(tf.expand_dims(end_tw,2))
        emb_begin_tw = self.emb_begin_tw(tf.expand_dims(begin_tw,2))
        projected_end_tw = self.project_end_tw(emb_end_tw)
        projected_begin_tw = self.project_end_tw(emb_begin_tw)

        # expanded_q,e: [batch_size x max_time x dim]
        e = self.project_ref(ref)
        q = self.project_query(query) #[batch_size x dim]
        expanded_q = tf.tile(tf.expand_dims(q,1),[1,max_time,1])

        # v_view:[batch_size x dim x 1]
        v_view = tf.tile( self.v, [tf.shape(e)[0],1,1])

        # u : [batch_size x max_time x dim] * [batch_size x dim x 1] =
        #       [batch_size x max_time]
        u = tf.squeeze(tf.matmul(self.tanh(expanded_q + e + d + projected_begin_tw + projected_end_tw), v_view),2)

        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u

        return e, logits
