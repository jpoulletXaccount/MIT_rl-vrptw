import tensorflow as tf
import numpy as np
import pickle,time,os

from configs import ParseParams
from shared.graph_embedding.useful_files.gnn_film_model import GNN_FiLM_Model
from shared.graph_embedding.useful_files.number_vehicle_task import Nb_Vehicles_Task
from VRP.vrp_utils import DataGenerator
# from VRPTW.vrptw_utils import DataGenerator

class Embedding(object):
    '''
    This class is the base class for embedding the input graph.
    '''
    def __init__(self,emb_type, embedding_dim):
        self.emb_type = emb_type
        self.embedding_dim = embedding_dim

        self.total_time = 0

    def __call__(self,input_pnt):
        # returns the embeded tensor. Should be implemented in child classes
        pass

class LinearEmbedding(Embedding):
    '''
    This class implements linear embedding. It is only a mapping 
    to a higher dimensional space.
    '''
    def __init__(self,embedding_dim,_scope=''):
        '''
        Input: 
            embedding_dim: embedding dimension
        '''

        super(LinearEmbedding,self).__init__('linear',embedding_dim)
        self.project_emb = tf.layers.Conv1D(embedding_dim,1,
            _scope=_scope+'Embedding/conv1d')

    def __call__(self,input_pnt):
        # emb_inp_pnt: [batch_size, max_time, embedding_dim]
        time_init = time.time()
        emb_inp_pnt = self.project_emb(input_pnt)
        # emb_inp_pnt = tf.Print(emb_inp_pnt,[emb_inp_pnt])
        self.total_time += time.time() - time_init
        return emb_inp_pnt


class GraphEmbedding(Embedding):
    """
    This class implement a graph embedding. The specificity is that it has already been optimized on
    another task. Implementation of transfer learning.
    """
    def __init__(self,args,data_test):
        assert args['embedding_dim'] == 30, args['embedding_dim']
        super(GraphEmbedding, self).__init__('graph',embedding_dim=args['embedding_dim'])

        self.n_nodes = args['n_nodes']
        self.embedding_dim =  args['embedding_dim']
        model_path = 'shared/graph_embedding/model_storage/' + args['task'] + '_model.pickle'
        result_dir = args['log_dir'] + '/embedding/'
        os.makedirs(result_dir)
        self.graph_model = self.restore(model_path, result_dir)
        self.graph_model.params['max_nodes_in_batch'] = args['test_size'] * self.n_nodes + 10 # We can process larger batches if we don't do training
        self.embedded_data = self(data_test)


    def __call__(self, input_data):
        """
        :param input_data: the input data as given by the env i.e. [batch_size x max_time x dim_task]
        :return: an embedding corresponding to the final node represenatation obtained via transfer learning.
        """
        time_init = time.time()
        embedded_data = self.graph_model.test(input_data)
        embedded_data = np.reshape(embedded_data,(-1,self.n_nodes,self.embedding_dim))
        self.total_time += time.time() - time_init

        return embedded_data

    @staticmethod
    def restore(saved_model_path: str, result_dir: str, run_id: str = None):
        print("Loading model from file %s." % saved_model_path)
        with open(saved_model_path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # model_cls, _ = name_to_model_class(data_to_load['model_class']({}))   # before...
        model_cls = GNN_FiLM_Model
        task_cls, additional_task_params = Nb_Vehicles_Task, {"data_kind":'transfer_learning'}

        if run_id is None:
            run_id = "_".join([task_cls.name(), model_cls.name(data_to_load['model_params']), time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])

        task = task_cls(data_to_load['task_params'])
        task.restore_from_metadata(data_to_load['task_metadata'])

        model = model_cls(data_to_load['model_params'], task, run_id, result_dir)
        model.load_weights(data_to_load['weights'])

        model.log_line("Loaded model from snapshot %s." % saved_model_path)

        return model


def test():
    args, prt = ParseParams()
    data_Gen = DataGenerator(args)
    # print(data_Gen.test_data)
    print(data_Gen.test_data.shape)

    graph_embedding = GraphEmbedding(args,data_Gen.test_data)
    data = data_Gen.get_train_next()
    graph_embedding(data)

