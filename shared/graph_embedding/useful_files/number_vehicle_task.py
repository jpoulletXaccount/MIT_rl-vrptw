
import tensorflow as tf
import numpy as np
from copy import deepcopy
from dpu_utils.utils import RichPath, LocalPath
from collections import namedtuple
import time

from typing import Any, Dict, Tuple, List, Iterable,Iterator
from .sparse_graph_task import Sparse_Graph_Task,DataFold,MinibatchData
from shared.graph_embedding import transfer_learning_dataset_utils


StopsData = namedtuple('StopsData', ['adj_lists','type_to_node_to_num_incoming_edges', 'num_stops', 'node_features', 'label'])


class Nb_Vehicles_Task(Sparse_Graph_Task):
    """
    Instancie une task de classification en nombre de vehicles
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # Things that will be filled once we load data:
        self.__num_edge_types = 5
        self.__initial_node_feature_size = 0
        self.__num_output_classes = 5

        # specific map from taks to helpers
        self._mapping = {'created_dataset': None,
                         'ups_dataset': None,
                         'transfer_learning':transfer_learning_dataset_utils.TransferDatasetUtils}
        self._true_dist = False
        if self._true_dist:
            self.__num_edge_types =1

    @classmethod
    def default_params(cls):
        """
        Applied to the class object, return the a list of specific param
        :return:
        """
        params = super().default_params()
        params.update({
            'add_self_loop_edges': True,
            'use_graph': True,
            'activation_function': "tanh",
            'out_layer_dropout_keep_prob': 0.8,
        })
        return params

    @staticmethod
    def name() -> str:
        return "Nb_Vehicles"

    @staticmethod
    def default_data_path() -> str:
        return "data/number_vehicles"


    def get_metadata(self) -> Dict[str, Any]:
        """
        :return: a dict with all the params related to the task
        """
        metadata = super().get_metadata()
        metadata['initial_node_feature_size'] = self.__initial_node_feature_size
        metadata['num_output_classes'] = self.__num_output_classes
        metadata['num_edge_types'] = self.__num_edge_types
        return metadata

    def restore_from_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        From a dict of parameters, restore it
        :param metadata:
        """
        super().restore_from_metadata(metadata)
        self.__initial_node_feature_size = metadata['initial_node_feature_size']
        self.__num_output_classes = metadata['num_output_classes']
        self.__num_edge_types = metadata['num_edge_types']

    @property
    def num_edge_types(self) -> int:
        return self.__num_edge_types

    @property
    def initial_node_feature_size(self) -> int:
        return self.__initial_node_feature_size

    # -------------------- Data Loading --------------------
    def load_data(self, path: RichPath) -> None:
        """
        Main function to load training and validation data
        :param path: the path to load the data
        """
        train_data, valid_data, test_data = self.__load_data(path)
        self._loaded_data[DataFold.TRAIN] = train_data
        self._loaded_data[DataFold.VALIDATION] = valid_data
        self._loaded_data[DataFold.TEST] = test_data

    def load_eval_data_from_path(self, path: RichPath) -> Iterable[Any]:
        data_path = self.default_data_path()
        helper_loader = self._mapping[self.params['data_kind']](data_path,self.num_edge_types, self._true_dist)
        dist_matrix,type_num, features, labels = helper_loader.fast_load_data_path(path)
        # all_dist_matrix,all_type_num, all_features, all_labels = helper_loader.load_data()
        self.__initial_node_feature_size = helper_loader.number_features
        self.__num_output_classes = helper_loader.number_labels

        test_data = self._process_raw_data(dist_matrix,type_num,features,labels)
        # test_data = self._process_raw_data(all_dist_matrix[DataFold.TEST],all_type_num[DataFold.TEST],all_features[DataFold.TEST],all_labels[DataFold.TEST])

        return test_data

    def load_eval_data_from_input(self, input) -> Iterable[Any]:
        helper_loader = self._mapping[self.params['data_kind']](input,self.num_edge_types, self._true_dist)
        dist_matrix,type_num, features, labels = helper_loader.fast_load_data_test(input)

        self.__initial_node_feature_size = helper_loader.number_features
        self.__num_output_classes = helper_loader.number_labels

        test_data = self._process_raw_data(dist_matrix,type_num,features,labels)

        return test_data

    def __load_data(self, data_directory: RichPath):
        assert isinstance(data_directory, LocalPath), "NumberVehiclesTask can only handle local data"
        data_path = data_directory.path
        print(" Loading NumberVehicles data from %s." % (data_path,))
        helper_loader = self._mapping[self.params['data_kind']](data_path,self.num_edge_types, self._true_dist)
        all_dist_matrix,all_type_num, all_features, all_labels = helper_loader.load_data()
        self.__initial_node_feature_size = helper_loader.number_features
        self.__num_output_classes = helper_loader.number_labels


        train_data = self._process_raw_data(all_dist_matrix[DataFold.TRAIN],all_type_num[DataFold.TRAIN],all_features[DataFold.TRAIN],all_labels[DataFold.TRAIN])
        valid_data = self._process_raw_data(all_dist_matrix[DataFold.VALIDATION],all_type_num[DataFold.VALIDATION],all_features[DataFold.VALIDATION],all_labels[DataFold.VALIDATION])
        test_data = self._process_raw_data(all_dist_matrix[DataFold.TEST],all_type_num[DataFold.TEST],all_features[DataFold.TEST],all_labels[DataFold.TEST])

        return train_data, valid_data, test_data

    def _process_raw_data(self,dist_matrix,type_num, features, labels):
        """
        Process the data to put it into right format
        :return: data under the form of lists of StopData
        """
        # processed_data = []
        # for i in range(0,len(labels)):
        #     processed_data.append(StopsData(adj_lists=dist_matrix[i],
        #                                     type_to_node_to_num_incoming_edges=type_num[i],
        #                                     num_stops=len(features[i,:,0]),
        #                                     node_features=features[i,:,:],
        #                                     label=labels[i]))
        #
        # return processed_data

        processed_data = []
        for i in range(0,len(features)):
            processed_data.append(StopsData(adj_lists=dist_matrix[i],
                                            type_to_node_to_num_incoming_edges=type_num[i],
                                            num_stops=len(features[i]),
                                            node_features=features[i],
                                            label=labels[i]))

        return processed_data


    def make_task_input_model(self,
                              placeholders: Dict[str, tf.Tensor],
                              model_ops: Dict[str, tf.Tensor],
                              ) -> None:
        """
        Create a task-specific input model. The default implementation
        simply creates placeholders to feed the input in, but more advanced
        variants could include sub-networks determining node features,
        for example.

        This method cannot assume the placeholders or model_ops dictionaries
        to be pre-populated, and needs to add at least the following
        entries to model_ops:
         * 'initial_node_features': float32 tensor of shape [V, D], where V
           is the number of nodes and D is the initial hidden dimension
           (needs to match the value of task.initial_node_feature_size).
         * 'adjacency_lists': list of L int32 tensors of shape [E, 2], where
           L is the number of edge types and E the number of edges of that
           type.
           Hence, adjacency_lists[l][e,:] == [u, v] means that u has an edge
           of type l to v.
         * 'type_to_num_incoming_edges': int32 tensor of shape [L, V], where
           L is the number of edge types and V the number of nodes.
           type_to_num_incoming_edges[l, v] = k indicates that node v has k
           incoming edges of type l.

        Arguments:
            placeholders: Dictionary of placeholders used by the model, to
                be extended with task-specific placeholders.
            model_ops: Dictionary of named operations in the model, to
                be extended with task-specific operations.
        """
        placeholders['initial_node_features'] = \
            tf.placeholder(dtype=tf.float32, shape=[None, self.__initial_node_feature_size], name='initial_node_features')
        placeholders['adjacency_lists'] = \
            [tf.placeholder(dtype=tf.float32, shape=[None, 3], name='adjacency_e%s' % e)
             for e in range(self.num_edge_types)]
        placeholders['type_to_num_incoming_edges'] = \
            tf.placeholder(dtype=tf.float32, shape=[self.num_edge_types, None], name='type_to_num_incoming_edges')

        model_ops['initial_node_features'] = placeholders['initial_node_features']
        model_ops['adjacency_lists'] = placeholders['adjacency_lists']
        model_ops['type_to_num_incoming_edges'] = placeholders['type_to_num_incoming_edges']


    def make_task_output_model(self,
                               placeholders: Dict[str, tf.Tensor],
                               model_ops: Dict[str, tf.Tensor],
                               ) -> None:
        """
        Create task-specific output model. For this, additional placeholders
        can be created, but will need to be filled in the
        make_minibatch_iterator implementation.

        This method may assume existence of the placeholders and ops created in
        make_task_input_model and of the following:
            model_ops['final_node_representations']: a float32 tensor of shape
                [V, D], which holds the final node representations after the
                GNN layers.
            placeholders['num_graphs']: a int32 scalar holding the number of
                graphs in this batch.
        Order of nodes is preserved across all tensors.

        This method has to define model_ops['task_metrics'] to a dictionary,
        from which model_ops['task_metrics']['loss'] will be used for
        optimization. Other entries may hold additional metrics (accuracy,
        MAE, ...).

        Arguments:
            placeholders: Dictionary of placeholders used by the model,
                pre-populated by the generic graph model values, and to
                be extended with task-specific placeholders.
            model_ops: Dictionary of named operations in the model,
                pre-populated by the generic graph model values, and to
                be extended with task-specific operations.
        """
        placeholders['labels'] = tf.placeholder(tf.int32,shape=[None], name='labels')

        placeholders['graph_nodes_list'] = \
            tf.placeholder(dtype=tf.int32, shape=[None], name='graph_nodes_list')

        placeholders['out_layer_dropout_keep_prob'] = \
            tf.placeholder_with_default(input=tf.constant(1.0, dtype=tf.float32),
                                        shape=[],
                                        name='out_layer_dropout_keep_prob')

        final_node_representations = \
            tf.nn.dropout(model_ops['final_node_representations'], rate= 1- placeholders['out_layer_dropout_keep_prob'])

        model_ops['final_representation'] = final_node_representations
        # # flatten steps
        # print(final_node_representations)
        # final_node_representations = tf.reshape(final_node_representations,shape=placeholders['graph_nodes_list'])
        # print("after reshape ", final_node_representations)
        # hidden_dim = 64
        # for i in range(0,4):
        #     final_node_representations = tf.keras.layers.Dense(units=hidden_dim,
        #                                                 use_bias=True,
        #                                                 activation=tf.nn.relu)(final_node_representations)

        output_label_logits = \
            tf.keras.layers.Dense(units=self.__num_output_classes,
                                  use_bias=False,
                                  activation=None,
                                  name="OutputDenseLayer",
                                  )(final_node_representations)  # Shape [nb_node, Classes]

        # Sum up all nodes per-graph
        per_graph_outputs = tf.unsorted_segment_sum(data=output_label_logits,
                                                    segment_ids=placeholders['graph_nodes_list'],
                                                    num_segments=tf.cast(placeholders['num_graphs'],dtype=tf.int32))

        correct_preds = tf.equal(tf.argmax(per_graph_outputs, axis=1, output_type=tf.int32),
                                 placeholders['labels'])

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=per_graph_outputs,
                                                                labels=placeholders['labels'])

        total_loss = tf.reduce_sum(losses)

        number_correct_preds = tf.reduce_sum(tf.cast(correct_preds,tf.float32))
        number_of_predictions = tf.cast(placeholders['num_graphs'],tf.float32)
        accuracy = number_correct_preds / number_of_predictions
        tf.summary.scalar('accuracy', accuracy)

        model_ops['task_metrics'] = {
            'loss': total_loss / number_of_predictions,
            'total_loss': total_loss,
            'accuracy': accuracy,
        }





    # def make_minibatch_iterator(self,
    #                             input_env: tf.Tensor,
    #                             data_fold: DataFold,
    #                             model_placeholders: Dict[str, tf.Tensor],
    #                             num_node,
    #                             ) -> Iterator[MinibatchData]:
    #     """
    #     Create minibatches for a sparse graph model, usually by flattening
    #     many smaller graphs into one large graphs of disconnected components.
    #     This should produce one epoch's worth of minibatches.
    #
    #     Arguments:
    #         input_env: the input from the agent, [batch_size x max_time x dim_task]
    #         data_fold: Fold of the loaded data to iterate over.
    #         model_placeholders: The placeholders of the model that need to be
    #             filled with data. Aside from the placeholders introduced by the
    #             task in make_task_input_model and make_task_output_model.
    #         num_node: the number of nodes of the instances
    #
    #     Returns:
    #         Iterator over MinibatchData values, which provide feed dicts
    #         as well as some batch statistics.
    #     """
    #     assert data_fold == DataFold.TEST,data_fold
    #     out_layer_dropout_keep_prob = 1.0
    #
    #     num_graphs = tf.shape(input_env)[0]
    #
    #     # assert num_graphs * num_node <= max_nodes_per_batch
    #     batch_node_features = tf.reshape(input_env,shape = [-1,tf.shape(input_env)[2]])
    #     batch_node_labels = tf.zeros_like(input_env[:,0,0])
    #
    #     print('Number of nodes ',num_node)
    #     graph_node_list = tf.range(0,num_graphs)
    #     graph_node_list = tf.reshape(graph_node_list,shape=[-1,1])
    #     graph_node_list = tf.tile(graph_node_list,multiples=[1,num_node])
    #     graph_node_list = tf.reshape(graph_node_list,shape=[-1])
    #
    #     batch_feed_dict = {
    #             model_placeholders['initial_node_features']: batch_node_features,
    #             model_placeholders['graph_nodes_list']: graph_node_list,
    #             model_placeholders['labels']:batch_node_labels,
    #             model_placeholders['out_layer_dropout_keep_prob']: out_layer_dropout_keep_prob,
    #         }
    #
    #     input_dist = input_env[:,:,:2]
    #     square_input = tf.reduce_sum(tf.square(input_dist), 2)
    #
    #     dist_matrix = tf.sqrt(tf.reshape(square_input,[-1,num_node,1]) - 2 * tf.matmul(input_dist,input_dist,False,True) + tf.reshape(square_input,[-1,1,num_node]))
    #
    #     list_num_incoming_ege = []
    #     not_masked = tf.ones_like(dist_matrix,dtype=tf.bool)
    #     num_edges = 0
    #     for i in range(self.num_edge_types):
    #         true_for_edge = tf.less(dist_matrix,self._scale[i])
    #         if i != 0:
    #             true_for_edge = tf.logical_and(not_masked,true_for_edge)
    #
    #         indices = tf.where(true_for_edge)
    #         print('indices ')
    #         print(indices)
    #         num_edges += tf.shape(indices)[0]
    #         not_masked = tf.logical_and(not_masked,tf.logical_not(true_for_edge))    # we update the mask. The only one not masked are the one wich
    #                                                                                 # were not and did not belong to the edge type
    #         batch_feed_dict[model_placeholders['adjacency_lists'][i]] = indices
    #
    #         true_for_edge = tf.cast(true_for_edge,dtype=tf.int32)
    #         print(tf.reduce_sum(true_for_edge,axis=1))
    #         list_num_incoming_ege.append(tf.reduce_sum(true_for_edge,axis=0))
    #
    #     num_incoming_edge = tf.concat(list_num_incoming_ege,axis=0)
    #     print('num incoming edge ')
    #     print(num_incoming_edge)
    #     batch_feed_dict[model_placeholders['type_to_num_incoming_edges']] = num_incoming_edge
    #
    #     yield MinibatchData(feed_dict=batch_feed_dict,
    #                         num_graphs=num_graphs,
    #                         num_nodes=num_node * num_graphs,
    #                         num_edges=num_edges)


    def make_minibatch_iterator(self,
                                data: Iterable[Any],
                                data_fold: DataFold,
                                model_placeholders: Dict[str, tf.Tensor],
                                max_nodes_per_batch: int,
                                ) -> Iterator[MinibatchData]:
        """
        Create minibatches for a sparse graph model, usually by flattening
        many smaller graphs into one large graphs of disconnected components.
        This should produce one epoch's worth of minibatches.

        Arguments:
            data: Data to iterate over, created by either load_data or
                load_eval_data_from_path.
            data_fold: Fold of the loaded data to iterate over.
            model_placeholders: The placeholders of the model that need to be
                filled with data. Aside from the placeholders introduced by the
                task in make_task_input_model and make_task_output_model.
            max_nodes_per_batch: Maximal number of nodes that can be packed
                into one batch.

        Returns:
            Iterator over MinibatchData values, which provide feed dicts
            as well as some batch statistics.
        """
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(data)
            out_layer_dropout_keep_prob = self.params['out_layer_dropout_keep_prob']
        else:
            out_layer_dropout_keep_prob = 1.0

        # Pack until we cannot fit more graphs in the batch
        num_graphs = 0
        while num_graphs < len(data):

            num_graphs_in_batch = 0
            batch_node_features = []
            batch_node_labels = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
            batch_type_to_num_incoming_edges = []
            batch_graph_nodes_list = []
            node_offset = 0

            while num_graphs < len(data) and node_offset + len(data[num_graphs].node_features) < max_nodes_per_batch:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(cur_graph.node_features)
                batch_node_features.extend(cur_graph.node_features)
                batch_graph_nodes_list.append(np.full(shape=[num_nodes_in_graph],
                                                      fill_value=num_graphs_in_batch,
                                                      dtype=np.int32))

                for i in range(self.num_edge_types):
                    selected_adj_list = deepcopy(cur_graph.adj_lists[i])
                    selected_adj_list[:,0] += node_offset
                    selected_adj_list[:,1] += node_offset
                    batch_adjacency_lists[i].append(selected_adj_list)

                batch_type_to_num_incoming_edges.append(np.array(cur_graph.type_to_node_to_num_incoming_edges))
                batch_node_labels.append(cur_graph.label)
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_feed_dict = {
                model_placeholders['initial_node_features']: np.array(batch_node_features),
                model_placeholders['type_to_num_incoming_edges']: np.concatenate(batch_type_to_num_incoming_edges, axis=1),
                model_placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                model_placeholders['labels']: np.array(batch_node_labels),
                model_placeholders['out_layer_dropout_keep_prob']: out_layer_dropout_keep_prob,
            }

            # Merge adjacency lists:
            num_edges = 0
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)

                num_edges += adj_list.shape[0]
                batch_feed_dict[model_placeholders['adjacency_lists'][i]] = adj_list


            yield MinibatchData(feed_dict=batch_feed_dict,
                                num_graphs=num_graphs_in_batch,
                                num_nodes=node_offset,
                                num_edges=num_edges)


    def early_stopping_metric(self,
                              task_metric_results: List[Dict[str, np.ndarray]],
                              num_graphs: int,
                              ) -> float:
        """
        Given the results of the task's metric for all minibatches of an
        epoch, produce a metric that should go down (e.g., loss). This is used
        for early stopping of training.

        Arguments:
            task_metric_results: List of the values of model_ops['task_metrics']
                (defined in make_task_model) for each of the minibatches produced
                by make_minibatch_iterator.
            num_graphs: Number of graphs processed in this epoch.

        Returns:
            Numeric value, where a lower value indicates more desirable results.
        """
        # Early stopping based on average loss:
        return np.sum([m['total_loss'] for m in task_metric_results]) / num_graphs

    def pretty_print_epoch_task_metrics(self,
                                        task_metric_results: List[Dict[str, np.ndarray]],
                                        num_graphs: int,
                                        ) -> str:
        """
        Given the results of the task's metric for all minibatches of an
        epoch, produce a human-readable result for the epoch (e.g., average
        accuracy).

        Arguments:
            task_metric_results: List of the values of model_ops['task_metrics']
                (defined in make_task_model) for each of the minibatches produced
                by make_minibatch_iterator.
            num_graphs: Number of graphs processed in this epoch.

        Returns:
            String representation of the task-specific metrics for this epoch,
            e.g., mean absolute error for a regression task.
        """
        # print("length of the metric ", len(task_metric_results))
        return "Acc: %.2f%%" % (np.mean([task_metric_results[i]['accuracy'] for i in range(0,len(task_metric_results))])  * 100,)
