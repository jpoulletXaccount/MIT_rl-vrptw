
import os

from evaluation.useful_stops import route,stop,stop_tw

class Benchmark(object):
    """
    Perform a benchmark of the results
    """
    def __init__(self,args,env,prt):
        self.args = args
        self.env = env
        self.prt = prt


    def perform_benchmark(self,list_eval):
        """
        Perform the benchmark amongs the list of eval
        :param list_eval: a list of type [beam-search, or-tools, greedy...]
        :return: write in the prt
        """

        dict_eval_route = dict()
        dict_eval_bench = dict()
        for ev in list_eval:
            dict_eval_route[ev] = self._build_routes(ev)
            dict_eval_bench[ev] = {'best_length' :0, 'best_vehicles':0,'total_length':0, 'number_vehicles':0}

        nb_routes = self.args['test_size']

        for i in range(0,nb_routes):
            list_length = []
            list_vehicles = []
            for ev in list_eval:
                route_considered = dict_eval_route[ev][i]
                tot_dist = route_considered.length
                nb_vehi = route_considered.vehicles
                list_length.append(tot_dist)
                list_vehicles.append(nb_vehi)
                dict_eval_bench[ev]['total_length'] += tot_dist
                dict_eval_bench[ev]['number_vehicles'] += nb_vehi

            # update best
            min_dist = min(list_length)
            min_vehi = min(list_vehicles)
            found_vehi = 0
            found_dist = 0
            for k,ev in enumerate(list_eval):
                if list_length[k] == min_dist:
                    dict_eval_bench[ev]['best_length'] += 1
                    found_dist +=1
                if list_vehicles[k] == min_vehi:
                    dict_eval_bench[ev]['best_vehicles'] +=1
                    found_vehi +=1
            assert found_vehi >= 1
            assert found_dist >= 1

        self.prt.print_out(str(dict_eval_bench))


    def _build_routes(self,ev):
        """
        Build the routes corresponding to eval
        :param ev: the name of the evaluation performed
        :return: a list of routes
        """
        list_routes = []
        # build task name and datafiles
        task = self.args['task_name']

        if self.args['ups']:
            task_name = '{}-ups-size-{}-len-{}-results-{}.txt'.format(task,self.args['test_size'], self.env.n_nodes,ev)
        else:
            task_name = '{}-size-{}-len-{}-results-{}.txt'.format(task,self.args['test_size'], self.env.n_nodes,ev)
        fname = os.path.join(self.args['log_dir'],'results', task_name)

        input_file =open(fname, 'r')

        line = input_file.readline()
        while line:
            if task == 'vrp':
                route_created = self._read_line(line)
            else:
                assert task == 'vrptw'
                route_created = self._read_line_tw(line)
                #assert route_created.check_feasible_tw(),ev
                if not route_created.check_feasible_tw():
                    print("route infeasible ", ev)
                    #assert False

            if route_created.nb_stops != self.env.n_nodes -1:
                print("pb in route ",ev, " number stop ", route_created.nb_stops)
            list_routes.append(route_created)
            line = input_file.readline()

        input_file.close()

        return list_routes


    def _read_line(self,line):
        """
        :param line:
        :return: a route in the case where no tw
        """
        assert self.args['task_name'] == 'vrp', self.args['task_name']
        words = line.strip().split(" ")
        nb_stops = int(len(words)/2)

        seq = []
        for i in range(0,nb_stops):
            new_guid = "i"
            seq.append(stop.Stop(guid=new_guid,
                                x= words[2*i],
                                y= words[2*i+1],
                                demand=-1))

        return route.Route(seq)


    def _read_line_tw(self,line):
        """
        :param line:
        :return: a route in the case where tw
        """
        assert self.args['task_name'] == 'vrptw', self.args['task_name']
        words = line.strip().split(" ")
        nb_stops = int(len(words)/4)

        seq = []
        for i in range(0,nb_stops):
            new_guid = "i"
            seq.append(stop_tw.StopTW(guid=new_guid,
                                        x= words[4*i],
                                        y= words[4*i+1],
                                      begin_tw= words[4*i +2],
                                      end_tw= words[4*i + 3],
                                    demand=-1))

        return route.Route(seq)
