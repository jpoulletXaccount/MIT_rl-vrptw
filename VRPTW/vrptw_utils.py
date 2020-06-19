import numpy as np
import tensorflow as tf
import os
import warnings
import collections



def create_VRPTW_dataset(
        n_problems,
        n_cust,
        data_dir,
        seed=None,
        data_type='train'):
    '''
    This function creates VRPTW instances and saves them on disk. If a file is already available,
    it will load the file.
    Input:
        n_problems: number of problems to generate.
        n_cust: number of customers in the problem.
        data_dir: the directory to save or load the file.
        seed: random seed for generating the data.
        data_type: the purpose for generating the data. It can be 'train', 'val', or any string.
    output:
        data: a numpy array with shape [n_problems x (n_cust+1) x 5]
        in the last dimension, we have x,y,begin_tw,end_tw,demand for customers. The last node is for depot and
        it has demand 0.
     '''

    # set random number generator
    n_nodes = n_cust +1     # 1 is for depot
    if seed == None:
        rnd = np.random
    else:
        rnd = np.random.RandomState(seed)

    # build task name and datafiles
    task_name = 'vrptw-size-{}-len-{}-{}.txt'.format(n_problems, n_nodes,data_type)
    fname = os.path.join(data_dir, task_name)

    # cteate/load data
    if os.path.exists(fname):
        print('Loading dataset for {}...'.format(task_name))
        data = np.loadtxt(fname,delimiter=' ')
        data = data.reshape(-1, n_nodes,5)
    else:
        print('Creating dataset for {}...'.format(task_name))
        # Generate a training set of size n_problems
        x = rnd.uniform(0,1,size=(n_problems,n_nodes,2))
        x = np.around(x, decimals=5)
        d = rnd.randint(1,10,[n_problems,n_nodes,1])

        # we set the day as being from 0 to 8.
        b_tw = 0
        e_tw = 8
        dur_min = 0.5
        dur_max = 1
        # Therefore, the begin time window is between 1 and 7, and its duration is from 0.5 to 1 ( from 1 since need to make sure that feasible, and 1.5 > sqrt(2))
        tw_begin = rnd.uniform(b_tw + dur_max,e_tw - dur_min *2, size=(n_problems,n_nodes,1))
        tw_end = tw_begin + rnd.uniform(dur_min, dur_max,size=(n_problems,n_nodes,1))

        d[:,-1]=0 # demand of depo is set to zero
        tw_begin[:,-1] = b_tw      # time window of the depo begin at 0
        tw_end[:,-1] = e_tw     # and ends at infinity (for the moment)

        data = np.concatenate([x,tw_begin,tw_end,d],axis =2)
        np.savetxt(fname, data.reshape(-1, n_nodes*5))

    return data


class DataGenerator(object):
    def __init__(self,
                 args):

        '''
        This class generates VRPTW problems for training and test
        Inputs:
            args: the parameter dictionary. It should include:
                args['random_seed']: random seed
                args['test_size']: number of problems to test
                args['n_nodes']: number of nodes
                args['n_cust']: number of customers
                args['batch_size']: batchsize for training
        '''
        self.args = args
        self.rnd = np.random.RandomState(seed= args['random_seed'])
        assert not self.args['ups']

        # create test data
        self.n_problems = args['test_size']
        self.test_data = create_VRPTW_dataset(self.n_problems,args['n_cust'],args['data_dir'],
            seed = args['random_seed']+1,data_type='test')

        self.reset()
        print('Created train iterator.')


    def reset(self):
        self.count = 0


    def get_train_next(self):
        '''
        Get next batch of problems for training
        Retuens:
            input_data: data with shape [batch_size x max_time x 5]
        '''

        x = self.rnd.uniform(0,1,size=(self.args['batch_size'],self.args['n_nodes'],2))
        d = self.rnd.randint(1,10,[self.args['batch_size'],self.args['n_nodes'],1])

        # we set the day as being from 0 to 8.
        b_tw = 0
        e_tw = 8
        dur_min = 0.5
        dur_max = 1
        # Therefore, the begin time window is between 1 and 7, and its duration is from 0.5 to 1 ( from 1 since need to make sure that feasible, and 1.5 > sqrt(2))
        tw_begin = self.rnd.uniform(b_tw + dur_max,e_tw - dur_min *2, size=(self.args['batch_size'],self.args['n_nodes'],1))
        tw_end = tw_begin + self.rnd.uniform(dur_min, dur_max,size=(self.args['batch_size'],self.args['n_nodes'],1))

        d[:,-1]=0 # demand of depo is set to zero
        tw_begin[:,-1] = b_tw      # time window of the depo begin at 0
        tw_end[:,-1] = e_tw        # and ends at infinity (for the moment)

        input_data = np.concatenate([x,tw_begin,tw_end,d],axis =2)

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
                                         "time",
                                         "demand",
                                         'd_sat',
                                         "mask"))):
    pass

class Env(object):
    def __init__(self,
                 args):
        '''
        This is the environment for VRPTW.
        Inputs:
            args: the parameter dictionary. It should include:
                args['n_nodes']: number of nodes in VRP
                args['n_custs']: number of customers in VRP
                args['input_dim']: dimension of the problem which is 5 (since we have added the tw)
        '''
        self.capacity = args['capacity']
        self.n_nodes = args['n_nodes']
        self.n_cust = args['n_cust']
        self.input_dim = args['input_dim']
        self.input_data = tf.placeholder(tf.float32,\
            shape=[None,self.n_nodes,self.input_dim])       # The dimension of the first (None) can be of any size

        self.embeded_data = tf.placeholder(tf.float32,shape=[None,self.n_nodes,args['embedding_dim']])
        self.input_data_norm = tf.placeholder(tf.float32,\
            shape=[None,self.n_nodes,self.input_dim])       # The dimension of the first (None) can be of any size
        self.input_pnt = self.input_data[:,:,:(self.input_dim -1)]  # all but demand
        self.demand = self.input_data[:,:,-1]
        self.all_x = self.input_data[:,:,0]
        self.all_y = self.input_data[:,:,1]
        self.all_b_tw = self.input_data[:,:,2]
        self.all_e_tw = self.input_data[:,:,3]
        self.previous_x = self.input_data[:,self.n_nodes -1,0]  # get the location x of all the depots
        self.previous_y = self.input_data[:,self.n_nodes -1,1]  # get the location y of all the depots

        self.batch_size = tf.shape(self.input_pnt)[0]

        # To be defined later on
        self.time = None
        self.load = None
        self.mask = None


    def reset(self,beam_width=1):
        '''
        Resets the environment. This environment might be used with different decoders.
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.
        '''
        # dimensions
        self.beam_width = beam_width
        self.batch_beam = self.batch_size * beam_width

        self.input_pnt = self.input_data[:,:,:(self.input_dim -1)]        # corresponds to all x,y,begin_tw, end_tw
        self.demand = self.input_data[:,:,-1]                             # corresponds to all the demand, sixe[batch,nb_nodes]
        self.all_x = self.input_data[:,:,0]
        self.all_y = self.input_data[:,:,1]
        self.all_b_tw = self.input_data[:,:,2]
        self.all_e_tw = self.input_data[:,:,3]
        self.previous_x = self.input_data[:,self.n_nodes -1,0]            # corresponds to the x of all the depots
        self.previous_x = tf.expand_dims(self.previous_x,1)               # dim[batch_size,1]
        self.previous_y = self.input_data[:,self.n_nodes -1,1]            # idem but for the y
        self.previous_y = tf.expand_dims(self.previous_y,1)               # dim[batch_size,1]

        # modify the self.input_pnt and self.demand for beam search decoder
#         self.input_pnt = tf.tile(self.input_pnt, [self.beam_width,1,1])

        # demand: [batch_size * beam_width, max_time]
        # demand[i] = demand[i+batchsize]
        self.demand = tf.tile(self.demand, [self.beam_width,1])
        self.all_x = tf.tile(self.all_x, [self.beam_width,1])
        self.all_y = tf.tile(self.all_y, [self.beam_width,1])
        self.all_b_tw = tf.tile(self.all_b_tw, [self.beam_width,1])
        self.all_e_tw = tf.tile(self.all_e_tw, [self.beam_width,1])
        self.previous_x = tf.tile(self.previous_x, [self.beam_width,1])
        self.previous_y = tf.tile(self.previous_y, [self.beam_width,1])

        # load: [batch_size * beam_width]
        self.load = tf.ones([self.batch_beam])*self.capacity
        #self.time = tf.expand_dims(tf.zeros([self.batch_beam]),1)
        self.time = tf.zeros([self.batch_beam])

        # create mask
        self.mask = tf.zeros([self.batch_size*beam_width,self.n_nodes],
                dtype=tf.float32)

        # update mask -- mask if customer demand is 0 and depot
        self.mask = tf.concat([tf.cast(tf.equal(self.demand,0), tf.float32)[:,:-1],
            tf.ones([self.batch_beam,1])],1)

        state = State(load=self.load,
                      time = self.time,
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
            #time:[batch_size*beam_width]
            self.time = tf.gather_nd(self.time,batchedBeamIdx)
            #MASK:[batch_size*beam_width x sourceL]
            self.mask = tf.gather_nd(self.mask,batchedBeamIdx)

            # Previous location [batch_size*beam_width x sourceL]
            self.previous_x = tf.gather_nd(self.previous_x,batchedBeamIdx)
            self.previous_y = tf.gather_nd(self.previous_y,batchedBeamIdx)

            self.all_x = tf.gather_nd(self.all_x,batchedBeamIdx)
            self.all_y = tf.gather_nd(self.all_y,batchedBeamIdx)
            self.all_b_tw = tf.gather_nd(self.all_b_tw, batchedBeamIdx)
            self.all_e_tw = tf.gather_nd(self.all_e_tw, batchedBeamIdx)


        BatchSequence = tf.expand_dims(tf.cast(tf.range(self.batch_beam), tf.int64), 1)
        batched_idx = tf.concat([BatchSequence,idx],1)

        # how much the demand is satisfied
        d_sat = tf.minimum(tf.gather_nd(self.demand,batched_idx), self.load)

        # update the demand
        d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(self.demand),tf.int64))      # sparse tensor containing d_sat for the interesting idx
        self.demand = tf.subtract(self.demand, d_scatter)

        # update load
        self.load -= d_sat

        # how much time has been spent
        visited_x = tf.gather_nd(self.all_x,batched_idx)
        visited_x = tf.expand_dims(visited_x,1)
        visited_y = tf.gather_nd(self.all_y,batched_idx)
        visited_y = tf.expand_dims(visited_y,1)
        t_spent = tf.sqrt(tf.square(self.previous_x - visited_x) + tf.square(self.previous_y - visited_y))
        t_spent = tf.squeeze(t_spent,[1])

        # update the previous location
        self.previous_x = visited_x
        self.previous_y = visited_y

        # update time, max of going there and wait
        self.time = tf.maximum(self.time + t_spent, tf.gather_nd(self.all_b_tw,batched_idx))

        # if in depot
        depot_flag = tf.squeeze(tf.cast(tf.equal(idx,self.n_cust),tf.float32),1)

        # refill the truck -- idx: [10,9,10] -> load_flag: [1 0 1]
        self.load = tf.multiply(self.load,1- depot_flag) + depot_flag *self.capacity
        # reset the time if in depot

        self.time = tf.multiply(self.time, 1- depot_flag)

        # mask for customers with zero demand (except depot, cf :-1)
        self.mask = tf.concat([tf.cast(tf.equal(self.demand,0), tf.float32)[:,:-1],
                                          tf.zeros([self.batch_beam,1])],1)

        # mask if load= 0
        # mask if in depot and there is still a demand

        self.mask += tf.concat( [tf.tile(tf.expand_dims(tf.cast(tf.equal(self.load,0),
            tf.float32),1), [1,self.n_cust]),
            tf.expand_dims(tf.multiply(tf.cast(tf.greater(tf.reduce_sum(self.demand,1),0),tf.float32),
                             tf.squeeze( tf.cast(tf.equal(idx,self.n_cust),tf.float32))),1)],1)

        # put the previous_x y as a matrix [batchsize * n_nodes]
        matrix_x = tf.tile(self.previous_x,[1,self.n_nodes])
        matrix_y = tf.tile(self.previous_y,[1,self.n_nodes])
        travel_time = tf.sqrt(tf.square(matrix_x - self.all_x) + tf.square(matrix_y - self.all_y))
        self.time = tf.expand_dims(self.time,1)
        arrival_time = tf.tile(self.time, [1,self.n_nodes]) + travel_time

        # mask for customers for which we will arrive after the end tw
        self.mask += tf.concat([tf.cast(tf.greater(arrival_time,self.all_e_tw), tf.float32)[:,:-1],
                                          tf.zeros([self.batch_beam,1])],1)

        self.time = tf.squeeze(self.time,[1])

        state = State(load=self.load,
                      time= self.time,
                    demand = self.demand,
                    d_sat = d_sat,
                    mask = self.mask )

        return state


def reward_func(sample_solution, decode_len=0.0, n_nodes=0.0, depot=None):
    """The reward for the VRP task is defined as the
    negative value of the route length

    Args:
        sample_solution : a list tensor of size decode_len of shape [batch_size x input_dim]
        depot: if not None, then means that we are aiming at decreasing the number of return to thde depot

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

    # make sure that we only take x,y (and not b_tw and e_tw)
    sample_solution = sample_solution[:,:,:2]

    sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1],0),
         sample_solution[:-1]),0)
    # get the reward based on the route lengths

    route_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow(\
        (sample_solution_tilted - sample_solution) ,2), 2) , .5), 0)

    if not depot is None:
        reward = tf.add(tf.scalar_mul(70.0,tf.scalar_mul(1.0/n_nodes,depot_visits)),tf.scalar_mul(30.0,tf.divide(route_lens_decoded,max_lens_decoded)))
        return reward
    else:
        return route_lens_decoded
