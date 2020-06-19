
import numpy as np
import tensorflow as tf

from configs import ParseParams


def load_task_specific_generator(task,ups):
    """
    This function load task-specific generators
    """
    if task == 'vrp':
        if ups:
            from UPS.vrp_ups_utils import create_VRP_UPS_dataset,Env
            dataset_creator = create_VRP_UPS_dataset
        else:
            from VRP.vrp_utils import create_VRP_dataset,Env
            dataset_creator = create_VRP_dataset

    elif task == 'vrptw':
        if ups:
            from UPS.vrptw_ups_utils import create_VRPTW_UPS_dataset,Env
            dataset_creator = create_VRPTW_UPS_dataset

        else:
            from VRPTW.vrptw_utils import create_VRPTW_dataset,Env
            dataset_creator = create_VRPTW_dataset

    else:
        raise Exception('Task is not implemented')

    return dataset_creator,Env


def load_task_specific_eval(task):
    """
    Load taks specific, dependign of tw or not
    """
    if task == 'vrp':
        from evaluation.eval_VRP import eval_google_or

        return eval_google_or.EvalGoogleOR

    elif task == 'vrptw':
        from evaluation.eval_VRPTW import eval_tw_google_or

        return eval_tw_google_or.EvalTWGoogleOR

    else:
        raise Exception('Task is not implemented')



def main(args,prt):
    """
    Main function, create a dataset and route it
    :param args: the arguments, particularly the routing task performed
    :param prt:
    :return:
    """
    # Create the dataset instances
    data_creator, Env = load_task_specific_generator(args['task_name'],args['ups'])
    env = Env(args)
    data_creator(args['test_size'],args['n_cust'],args['data_dir'],
            seed = args['random_seed']+1,data_type='test')

    router = load_task_specific_eval(args['task_name'])
    object_eval = router(args,env,prt,args['min_trucks'])
    object_eval.perform_routing_transfer_learning()



if __name__ == '__main__':
    args, prt = ParseParams()
    args['test_size'] = 5000

    # Random
    random_seed = args['random_seed']
    if random_seed is not None and random_seed > 0:
        prt.print_out("# Set random seed to %d" % random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
    tf.reset_default_graph()

    main(args, prt)
