import numpy as np
import tensorflow as tf
import os
import warnings
import collections
import joblib


def create_VRP_UPS_dataset(
        n_problems,
        n_cust,
        data_dir,
        generator,
        data_type='train'):
    '''
    This function creates VRP instances and saves them on disk. If a file is already available,
    it will load the file.
    Input:
        n_problems: number of problems to generate.
        n_cust: number of customers in the problem.
        data_dir: the directory to save or load the file.
        seed: random seed for generating the data.
        data_type: the purpose for generating the data. It can be 'train', 'val', or any string.
    output:
        data: a numpy array with shape [n_problems x (n_cust+1) x 3]
        in the last dimension, we have x,y,demand for customers. The last node is for depot and
        it has demand 0.
     '''

    # set random number generator
    n_nodes = n_cust +1

    # build task name and datafiles
    task_name = 'vrp-ups-size-{}-len-{}-{}.txt'.format(n_problems, n_nodes,data_type)
    fname = os.path.join(data_dir, task_name)

     # cteate/load data
    if os.path.exists(fname):
        print('Loading dataset for {}...'.format(task_name))
        data = np.loadtxt(fname,delimiter=' ')
        data = data.reshape(-1, n_nodes,3)

    else:
         # Generate a training set of size n_problems
        data = []
        intfunct = np.vectorize(lambda x : max(1,np.round(x)))

        depot = [0,0,0]

        for i in range(0,n_problems):
            sample_X,_ = generator.sample(n_samples = n_cust)
            sample_X[:,2] = intfunct(sample_X[:,2])

            # concatenate depot
            final_data = np.append(sample_X,[depot],axis=0)
            data.append(final_data)

        data = np.array(data)
        np.savetxt(fname, data.reshape(-1, n_nodes*3))

        print("New test set created")

    return data


class DataGenerator(object):
    def __init__(self,
                 args):

        '''
        This class generates VRP problems for training and test
        Inputs:
            args: the parameter dictionary. It should include:
                args['random_seed']: random seed
                args['test_size']: number of problems to test
                args['n_nodes']: number of nodes
                args['n_cust']: number of customers
                args['batch_size']: batchsize for training
        '''
        self.args = args
        assert self.args['ups']
        path_gaussian = os.path.join('gaussian_mixture','cvrp.joblib')
        self.gaussian_generator = joblib.load(path_gaussian)

        self.n_problems = args['test_size']
        self.test_data = create_VRP_UPS_dataset(self.n_problems,args['n_cust'],args['data_dir'],
            generator=self.gaussian_generator,data_type='test')

        self.reset()



    def reset(self):
        self.count = 0

    def get_train_next(self):
        '''
        Get next batch of problems for training
        Retuens:
            input_data: data with shape [batch_size x max_time x 3]
        '''
        intfunct = np.vectorize(lambda x : max(1,np.round(x)))

        input_data = []
        depot = [0,0,0]

        for i in range(0,self.args['batch_size']):
            sample_X,_ = self.gaussian_generator.sample(n_samples = self.args['n_nodes']-1)
            sample_X[:,2] = intfunct(sample_X[:,2])

            # concatenate depot
            final_data = np.append(sample_X,[depot],axis=0)
            input_data.append(final_data)

        return input_data


    def get_test_next(self):
        '''
        Get next batch of problems for testing
        '''
        if self.count<self.args['test_size']:
            input_pnt = self.test_data[self.count:self.count+1]
            self.count +=1
        else:
            warnings.warn("The test iterator reset.")
            self.count = 0
            input_pnt = self.test_data[self.count:self.count+1]
            self.count +=1

        return input_pnt

    def get_test_all(self):
        '''
        Get all test problems
        '''
        return self.test_data


class State(collections.namedtuple("State",
                                        ("load",
                                         "demand",
                                         'd_sat',
                                         "mask"))):
    pass

class Env(object):
    def __init__(self,
                 args):
        '''
        This is the environment for VRP.
        Inputs:
            args: the parameter dictionary. It should include:
                args['n_nodes']: number of nodes in VRP
                args['n_custs']: number of customers in VRP
                args['input_dim']: dimension of the problem which is 3
        '''
        self.capacity = args['capacity']
        self.n_nodes = args['n_nodes']
        self.n_cust = args['n_cust']
        self.input_dim = args['input_dim']
        self.input_data = tf.placeholder(tf.float32,\
            shape=[None,self.n_nodes,self.input_dim])       # The dimension of the first (None) can be of any size

        self.input_pnt = self.input_data[:,:,:2]
        self.demand = self.input_data[:,:,-1]
        self.batch_size = tf.shape(self.input_pnt)[0]

    def reset(self,beam_width=1):
        '''
        Resets the environment. This environment might be used with different decoders.
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.
        '''

        # dimensions
        self.beam_width = beam_width
        self.batch_beam = self.batch_size * beam_width

        self.input_pnt = self.input_data[:,:,:2]        # corresponds to all x,y
        self.demand = self.input_data[:,:,-1]           # corresponds to all the demand, sixe[batch,nb_nodes]

        # modify the self.input_pnt and self.demand for beam search decoder
#         self.input_pnt = tf.tile(self.input_pnt, [self.beam_width,1,1])

        # demand: [batch_size * beam_width, max_time]
        # demand[i] = demand[i+batchsize]
        self.demand = tf.tile(self.demand, [self.beam_width,1])

        # load: [batch_size * beam_width]
        self.load = tf.ones([self.batch_beam])*self.capacity

        # create mask
        self.mask = tf.zeros([self.batch_size*beam_width,self.n_nodes],
                dtype=tf.float32)

        # update mask -- mask if customer demand is 0 and depot
        self.mask = tf.concat([tf.cast(tf.equal(self.demand,0), tf.float32)[:,:-1],
            tf.ones([self.batch_beam,1])],1)

        state = State(load=self.load,
                    demand = self.demand,
                    d_sat = tf.zeros([self.batch_beam,self.n_nodes]),
                    mask = self.mask )

        return state

    def step(self,
             idx,
             beam_parent=None):
        '''
        runs one step of the environment and updates demands, loads and masks
        '''

        # if the environment is used in beam search decoder
        if beam_parent is not None:
            # BatchBeamSeq: [batch_size*beam_width x 1]
            # [0,1,2,3,...,127,0,1,...],
            batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(self.batch_size), tf.int64),
                                                 [self.beam_width]),1)
            # batchedBeamIdx:[batch_size*beam_width]
            batchedBeamIdx= batchBeamSeq + tf.cast(self.batch_size,tf.int64)*beam_parent
            # demand:[batch_size*beam_width x sourceL]
            self.demand= tf.gather_nd(self.demand,batchedBeamIdx)
            #load:[batch_size*beam_width]
            self.load = tf.gather_nd(self.load,batchedBeamIdx)
            #MASK:[batch_size*beam_width x sourceL]
            self.mask = tf.gather_nd(self.mask,batchedBeamIdx)


        BatchSequence = tf.expand_dims(tf.cast(tf.range(self.batch_beam), tf.int64), 1)
        batched_idx = tf.concat([BatchSequence,idx],1)

        # how much the demand is satisfied
        d_sat = tf.minimum(tf.gather_nd(self.demand,batched_idx), self.load)

        # update the demand
        d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(self.demand),tf.int64))      # sparse tensor containing d_sat for the interesting idx
        self.demand = tf.subtract(self.demand, d_scatter)

        # update load
        self.load -= d_sat

        # refill the truck -- idx: [10,9,10] -> load_flag: [1 0 1]
        load_flag = tf.squeeze(tf.cast(tf.equal(idx,self.n_cust),tf.float32),1)
        self.load = tf.multiply(self.load,1-load_flag) + load_flag *self.capacity

        # mask for customers with zero demand
        self.mask = tf.concat([tf.cast(tf.equal(self.demand,0), tf.float32)[:,:-1],
                                          tf.zeros([self.batch_beam,1])],1)

        # mask if load= 0
        # mask if in depot and there is still a demand

        self.mask += tf.concat( [tf.tile(tf.expand_dims(tf.cast(tf.equal(self.load,0),
            tf.float32),1), [1,self.n_cust]),
            tf.expand_dims(tf.multiply(tf.cast(tf.greater(tf.reduce_sum(self.demand,1),0),tf.float32),
                             tf.squeeze( tf.cast(tf.equal(idx,self.n_cust),tf.float32))),1)],1)

        state = State(load=self.load,
                    demand = self.demand,
                    d_sat = d_sat,
                    mask = self.mask )

        return state

def reward_func(sample_solution, decode_len=0.0, n_nodes=0.0, depot=None):
    """The reward for the VRP task is defined as the
    negative value of the route length

    Args:
        sample_solution : a list tensor of size decode_len of shape [batch_size x input_dim]
        demands satisfied: a list tensor of size decode_len of shape [batch_size]

    Returns:
        rewards: tensor of size [batch_size]

    Example:
        sample_solution = [[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]]
        sourceL = 3
        batch_size = 2
        input_dim = 2
        sample_solution_tilted[ [[5,5]
                                                    #  [6,6]]
                                                    # [[1,1]
                                                    #  [2,2]]
                                                    # [[3,3]
                                                    #  [4,4]] ]
    """
    # make init_solution of shape [sourceL x batch_size x input_dim]
    if depot != None:
        counter = tf.zeros_like(depot)[:,0]
        depot_visits = tf.cast(tf.equal(sample_solution[0], depot), tf.float32)[:,0]
        for i in range(1,len(sample_solution)):
            interm_depot = tf.cast(tf.equal(sample_solution[i], depot), tf.float32)[:,0]
            counter = tf.add(tf.multiply(counter,interm_depot), interm_depot)
            depot_visits = tf.add(depot_visits, tf.multiply(interm_depot, tf.cast(tf.less(counter,1.5), tf.float32)))
            # depot_visits = tf.add(depot_visits,tf.cast(tf.equal(sample_solution[i], depot), tf.float32)[:,0])

        max_length = tf.stack([depot for d in range(decode_len)],0)
        max_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow(\
            (max_length - sample_solution) ,2), 2) , .5), 0)

    # make sample_solution of shape [sourceL x batch_size x input_dim]
    sample_solution = tf.stack(sample_solution,0)

    sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1],0),
         sample_solution[:-1]),0)
    # get the reward based on the route lengths
    route_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow(\
        (sample_solution_tilted - sample_solution) ,2), 2) , .5), 0)

    if depot != None:
        reward = tf.add(tf.scalar_mul(70.0,tf.scalar_mul(1.0/n_nodes,depot_visits)),tf.scalar_mul(30.0,tf.divide(route_lens_decoded,max_lens_decoded)))
        return reward
    else:
        return route_lens_decoded
