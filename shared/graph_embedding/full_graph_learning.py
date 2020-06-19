import tensorflow as tf
import numpy as np
import time

from shared.embeddings import Embedding
from shared.graph_embedding.useful_files.utils import get_activation
from shared.graph_embedding.useful_files.gnn_film import sparse_gnn_film_layer

class FullGraphEmbedding(Embedding):
    """
    Implements a graph embedding, not test
    """
    def __init__(self,embedding_dim,args):
        assert args['embedding_dim'] == 30, args['embedding_dim']
        super(FullGraphEmbedding,self).__init__('full_graph',embedding_dim)

        self.nb_feat = args['input_dim']
        self.n_nodes = args['n_nodes']

        self._scale = [5,12,25,50,100]
        self._scale = [i * np.sqrt(2)/100 for i in self._scale]     # rescale to the square

        self.drop_out = tf.placeholder(tf.float32,name='embedder_graph_dropout')
        self.params = {
            'graph_num_layers': 8,
            'graph_num_timesteps_per_layer': 3,

            'graph_layer_input_dropout_keep_prob': 0.8,
            'graph_dense_between_every_num_gnn_layers': 1,
            'graph_model_activation_function': 'tanh',
            'graph_residual_connection_every_num_layers': 1,
            'graph_inter_layer_norm': False,
            "hidden_size": 30,
            "graph_activation_function": "ReLU",
            "message_aggregation_function": "sum",
            "normalize_messages_by_num_incoming": True
            }


    def _propagate_graph_model(self,initial_node_features, incoming_edge, list_pair_adjancy):
        """
        Build the propagation model via graph
        :param initial_node_features:
        :param incoming_edge:
        :param list_pair_adjancy:
        :return:
        """
        h_dim= self.params['hidden_size']
        activation_fn = get_activation(self.params['graph_model_activation_function'])

        projected_node_features = tf.keras.layers.Dense(units=h_dim,
                                      use_bias=False,
                                      activation=activation_fn,
                                      )(initial_node_features)

        cur_node_representations = projected_node_features
        last_residual_representations = tf.zeros_like(cur_node_representations)
        for layer_idx in range(self.params['graph_num_layers']):
            # with tf.variable_scope('gnn_layer_%i' % layer_idx):
            cur_node_representations = \
                tf.nn.dropout(cur_node_representations, rate= 1- self.drop_out)
            if layer_idx % self.params['graph_residual_connection_every_num_layers'] == 0:
                t = cur_node_representations
                if layer_idx > 0:
                    cur_node_representations += last_residual_representations
                    cur_node_representations /= 2
                last_residual_representations = t
            cur_node_representations = \
                self._apply_gnn_layer(cur_node_representations,list_pair_adjancy,incoming_edge,self.params['graph_num_timesteps_per_layer'])
            if self.params['graph_inter_layer_norm']:
                cur_node_representations = tf.contrib.layers.layer_norm(cur_node_representations)
            if layer_idx % self.params['graph_dense_between_every_num_gnn_layers'] == 0:
                cur_node_representations = \
                    tf.keras.layers.Dense(units=h_dim,
                                          use_bias=False,
                                          activation=activation_fn,
                                          name="Dense",
                                          )(cur_node_representations)

        return cur_node_representations


    def _apply_gnn_layer(self,node_representations,adjacency_lists,type_to_num_incoming_edges,num_timesteps):
        """
        Apply the actual gnn layer
        """
        return sparse_gnn_film_layer(
            node_embeddings=node_representations,
            adjacency_lists=adjacency_lists,
            type_to_num_incoming_edges=type_to_num_incoming_edges,
            state_dim=self.params['hidden_size'],
            num_timesteps=num_timesteps,
            activation_function=self.params['graph_activation_function'],
            message_aggregation_function=self.params['message_aggregation_function'],
            normalize_by_num_incoming=self.params["normalize_messages_by_num_incoming"])


    def _prepare_input_data(self, input_tf):
        """
        Prepare the input data so that they are at the right size
        :param input_tf:
        :return:
        """

        batch_features = tf.reshape(input_tf,[-1,self.nb_feat])

        input_dist = input_tf[:,:,:2]
        square_input = tf.reduce_sum(tf.square(input_dist), 2)
        row = tf.reshape(square_input, [-1,self.n_nodes,1])
        col= tf.reshape(square_input,[-1,1,self.n_nodes])
        dist_matrix = tf.sqrt(tf.maximum(row - 2 * tf.matmul(input_dist,input_dist,False,True) + col,0.0))

        list_pair_edge = []
        list_num_incoming_ege = []
        not_masked = tf.ones_like(dist_matrix,dtype=tf.bool)
        tf.matrix_set_diag(not_masked,tf.zeros_like(not_masked[0,:,:]))

        for i in range(len(self._scale)):
            true_for_edge = tf.less_equal(dist_matrix,self._scale[i])
            true_for_edge = tf.logical_and(not_masked,true_for_edge)

            indices = tf.cast(tf.where(true_for_edge),dtype=tf.int32)
            offset = self.n_nodes * indices[:,0]    # get all batch value
            offset = tf.expand_dims(offset,axis=1)
            offset = tf.tile(offset,[1,2])
            true_indices_nodes = offset + indices[:,1:3]
            list_pair_edge.append(true_indices_nodes)

            num_incoming = tf.reduce_sum(tf.cast(true_for_edge,dtype=tf.int32), 1)
            num_incoming = tf.squeeze(tf.reshape(num_incoming,[1,-1]),0)

            list_num_incoming_ege.append(tf.cast(num_incoming,dtype=tf.float32))

            # update the mask
            not_masked = tf.logical_and(not_masked,tf.logical_not(true_for_edge)) # we update the mask. The only one not masked are the one wich
                                                                                    # were not and did not belong to the edge type
        final_incoming_edge = tf.stack(list_num_incoming_ege)


        return batch_features, final_incoming_edge, list_pair_edge

    def __call__(self, input_tf):
        """
        return the node embedding
        :param input_tf: the tensor corresponding to the embedding
        :return: a tensor
        """
        time_init = time.time()
        initial_node_features, incoming_edge, list_pair_adjancy = self._prepare_input_data(input_tf)

        final_node_representations = self._propagate_graph_model(initial_node_features,incoming_edge,list_pair_adjancy)
        final_node_representations = tf.reshape(final_node_representations,[-1,self.n_nodes,self.embedding_dim])

        self.total_time += time.time() - time_init

        return final_node_representations

