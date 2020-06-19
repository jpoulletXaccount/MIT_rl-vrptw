
import os,time
import numpy as np
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from evaluation.useful_stops import managerStops,route

class RoutingEvaluator(object):
    """
    Mother class of all the benchmarks method
    """

    def __init__(self,args,env,prt,min_veh):
        self.args = args
        self.env = env
        self.prt = prt
        self.min_veh = min_veh
        self.name = "mother_eval"

        self.output_file = None
        self.output_file_transfer_learning = None

    def _update_filename(self,benchmark_type):
        """
        :return: the name of the output file
        """
         # build task name and datafiles
        if self.args['ups']:
            task_name = 'vrp-ups-size-{}-len-{}-results-{}.txt'.format(self.args['test_size'], self.env.n_nodes,benchmark_type)
        else:
            task_name = 'vrp-size-{}-len-{}-results-{}.txt'.format(self.args['test_size'], self.env.n_nodes,benchmark_type)
        fname = os.path.join(self.args['log_dir'],'results', task_name)
        fname_transfer_learning = os.path.join(self.args['data_dir'],task_name)

        self.output_file = fname
        self.output_file_transfer_learning = fname_transfer_learning

    def _load_manager_stops(self):
        """
        Load all the manager stops required
        :return: a list of manager stops
        """
        list_manager = []

        # file
        if self.args['ups']:
            data_name = 'vrp-ups-size-{}-len-{}-test.txt'.format(self.args['test_size'], self.env.n_nodes)
        else:
            data_name = 'vrp-size-{}-len-{}-test.txt'.format(self.args['test_size'], self.env.n_nodes)
        data_loc = os.path.join(self.args['data_dir'], data_name)

        data_file =open(data_loc, 'r')
        line = data_file.readline()
        while line:
            list_manager.append(managerStops.managerStops.from_line(line))
            line = data_file.readline()

        data_file.close()
        return list_manager


    def _check_manager_stop(self,list_manager_stop):
        """
        Check that every stop fit the capacity of the truck, if not then divide it into two parts
        :return: nothing but update the list_manager_stop
        """
        cap = self.env.capacity
        for manager in list_manager_stop:
            manager.check_capacity(cap)


    def perform_routing(self):
        """
        Main function of the class, perform the routing on all the tests sets
        Write result in a new file
        """
        list_manager = self._load_manager_stops()
        self._check_manager_stop(list_manager)
        list_results = []
        time_beg = time.time()
        for manager in tqdm(list_manager):
            route_created = self._route_creator(manager)
            list_results.append(route_created)

        time_end = time.time() - time_beg
        self.prt.print_out("Finished evalution with " + str(self.name) + " in " + str(time_end))

        # output all the routes
        self._dump_results(list_results)


    def perform_routing_transfer_learning(self):
        """
        Main function of the class, perform the routing on all the tests sets
        Write result in a new file
        """
        list_manager = self._load_manager_stops()
        # list_manager = self._cluster_for_transfer_learning(list_manager)
        self._check_manager_stop(list_manager)
        list_results = []
        time_beg = time.time()
        for manager in tqdm(list_manager):
            route_created= self._route_creator(manager)
            list_results.append(route_created)

        time_end = time.time() - time_beg

        # output all the routes
        self._dump_results_for_transfer_learning(list_results)
        self.prt.print_out("Finished routing with " + str(self.name) + " in " + str(time_end) + " for " + str(len(list_results)) + " datapoints")


    def _dump_results(self,list_routes):
        """
        Write the results in the output file
        :param list_routes: a list of sequence of stops including depot if necessary
        """
        data_file =open(self.output_file, 'w')
        for seq in list_routes:
            for stop in seq:
                data_file.write(str(stop.x) + " " + str(stop.y) + " ")

            data_file.write("\n")

        data_file.close()


    def _dump_results_for_transfer_learning(self,list_routes):
        """
        Write the results in the output file
        :param list_routes: a list of sequence of stops including depot if necessary
        """
        data_file =open(self.output_file_transfer_learning, 'w')
        for seq in list_routes:
            route_considered = route.Route(seq)
            data_file.write(str(route_considered.vehicles) + "*")
            depot = route_considered.depot
            # write the depot
            data_file.write(str(depot.x) + "-" + str(depot.y) + "*" )
            stop_in_route = route_considered.get_only_true_stops()
            # assert len(stop_in_route) == route_considered.nb_stops
            for stop in stop_in_route:
                data_file.write(str(stop.x) + "-" + str(stop.y) + "-" + str(stop.demand) + "_")

            data_file.write("\n")

        data_file.close()


    def _cluster_for_transfer_learning(self,list_manager):
        """
        Perform a clustering steps on the current stops, so that we can get smaller vehicles
        :param list_manager: a list of manager stops
        :return: a new list of managers
        """
        new_list_manager = []
        cap = self.env.capacity
        for manager in list_manager:
            new_list_manager.append(manager)
            total_demand = manager.demand
            list_K = [int(total_demand/(1 * cap)), int(total_demand/(2 * cap)),int(total_demand/(3 * cap))]
            for k in list_K:
                if k != 0:
                    new_list_manager.extend(self.kmean_clustering(manager,k))

        return new_list_manager


    @staticmethod
    def kmean_clustering(manager_stop,number_cluster):
        """
        Perform a Kmean algorithm on the given stops with a given number of clusters
        :param manager_stop: a manger_stop
        :param number_cluster: the number of clusters
        :return: a list of manager
        """
        final_list_manager = []
        matrix_array = np.array([[stop.x,stop.y] for stop in manager_stop.values()])

        clusters_list_stop = KMeans(n_clusters = number_cluster).fit_predict(X=matrix_array)
        dict_cluster_stops = dict()
        for i,stop_id in enumerate(list(manager_stop.keys())):
            cluster_id = clusters_list_stop[i]

            if not cluster_id in dict_cluster_stops.keys():
                dict_cluster_stops[cluster_id] = []
            dict_cluster_stops[cluster_id].append(manager_stop[stop_id])

        for list_stop in dict_cluster_stops.values():
            final_list_manager.append(managerStops.managerStops.from_stops_list(manager_stop,list_stop))

        return final_list_manager


    def _route_creator(self,manager):
        """
        Route the manager, need to be overwritten
        :param manager: a stop manager
        :return:a route object
        """
        raise Exception("Need to be overwritten in children classes")



