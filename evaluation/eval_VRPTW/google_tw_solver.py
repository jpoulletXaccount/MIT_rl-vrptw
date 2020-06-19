
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

class GoogleSolverTW(object):
    """
    Solver of google tw or
    """

    def __init__(self,manager_stop_tw,env, min_veh):
        self.manager_stop_tw = manager_stop_tw
        self.env = env
        self.min_veh = min_veh
        self.dict_idx_guid = self._create_dict_idx_guid()

        self.multiplier = 10000       # Since google or tool takes only integer


    def _create_dict_idx_guid(self):
        """
        Match the stop.guid with an index for the solver
        :return: a dict[idx] = guid
        """
        dict_idx = dict()
        dict_idx[0] = self.manager_stop_tw.depot.guid

        comp = 1
        for stop_guid in self.manager_stop_tw:
            dict_idx[comp] = stop_guid
            comp +=1

        return dict_idx


    def create_data_model(self):
        """Stores the data for the problem."""
        data = dict()
        dist_matrix = [[0.0 for i in self.dict_idx_guid] for j in self.dict_idx_guid]
        for i in self.dict_idx_guid:
            if i == 0:
                stop_chosen = self.manager_stop_tw.depot
            else:
                stop_chosen = self.manager_stop_tw[self.dict_idx_guid[i]]
            for j in self.dict_idx_guid:
                if j == 0:
                    stop_tried = self.manager_stop_tw.depot
                else:
                    stop_tried = self.manager_stop_tw[self.dict_idx_guid[j]]
                dist_matrix[i][j] = self.multiplier * stop_chosen.get_dist_another_stop(stop_tried)        # we multiplied by 1000 because
                                                                                                # google or takes integer as input

        data['distance_matrix'] = dist_matrix
        list_tw = [(self.multiplier * self.manager_stop_tw.depot.begin_tw, self.multiplier * (self.manager_stop_tw.depot.end_tw +10))]
        list_tw+= [ ( self.multiplier * self.manager_stop_tw[self.dict_idx_guid[i]].begin_tw, self.multiplier * self.manager_stop_tw[self.dict_idx_guid[i]].end_tw) for i in self.dict_idx_guid if i !=0]
        data['time_windows'] =list_tw

        list_demand = [0] + [self.manager_stop_tw[self.dict_idx_guid[i]].demand for i in self.dict_idx_guid if i != 0]
        data['demands'] = list_demand

        data['num_vehicles'] = self.env.n_nodes
        data['vehicle_capacities'] = [self.env.capacity for i in range(0,self.env.n_nodes)]
        data['depot'] = 0

        return data


    def solve(self):
        """Solve the CVRP problem."""
        # Instantiate the data problem.
        data =self.create_data_model()

        # Create the routing index manager.
        manager_model = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager_model)

        # Add time windows constraints
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager_model.IndexToNode(from_index)
            to_node = manager_model.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

        routing.AddDimension(
            time_callback_index,
            self.multiplier * 100,  # waiting time
            self.multiplier * int(self.manager_stop_tw.depot.end_tw),  # vehicle maximum time
            True,  # start cumul to zero
            'Time')

        time_dimension = routing.GetDimensionOrDie('Time')
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == 0:
                continue
            index = manager_model.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(int(time_window[0]), int(time_window[1]))
        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(int(data['time_windows'][0][0]),
                                                    int(data['time_windows'][0][1]))

        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager_model.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # Allow to drop nodes.
        penalty = self.multiplier * 100000
        for node in range(1, len(data['distance_matrix'])):
            routing.AddDisjunction([manager_model.NodeToIndex(node)], penalty)

        penalty_veh = self.multiplier * 500     # would need to put 500 nodes in a vehicle to balance
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i)))
            if self.min_veh:
                routing.SetFixedCostOfVehicle(penalty_veh, i)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.time_limit.seconds = 10
        # Solve the problem.
        assignment = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if assignment:
            return self._parse_solution(data, manager_model,routing, assignment)

        else:
            print("Infeasible in Google OR")
            assert False


    def _parse_solution(self,data,manager_model,routing, assignment):
        """
        Parse solution on a form interpretable
        :return: a sequence of stops, including depot
        """
        seq_stop = [self.manager_stop_tw.depot]
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)

            nb_stop = 0
            while not routing.IsEnd(index):
                node_index = manager_model.IndexToNode(index)
                if nb_stop >0:
                    stop_guid = self.dict_idx_guid[node_index]
                    seq_stop.append(self.manager_stop_tw[stop_guid])
                index = assignment.Value(routing.NextVar(index))
                nb_stop +=1

            # add depot at the end if vehicles
            if nb_stop >1:
                seq_stop.append(self.manager_stop_tw.depot)

        return seq_stop
