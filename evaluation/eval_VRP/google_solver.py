
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

class GoogleSolver(object):
    """
    Solver of google or
    """

    def __init__(self,manager_stop,env,min_veh):
        self.manager_stop = manager_stop
        self.env= env
        self.min_veh = min_veh
        self.dict_idx_guid = self._create_dict_idx_guid()

        self.multiplier = 10000       # Since google or tool takes only integer


    def _create_dict_idx_guid(self):
        """
        Match the stop.guid with an index for the solver
        :return: a dict[idx] = guid
        """
        dict_idx = dict()
        dict_idx[0] = self.manager_stop.depot.guid

        comp = 1
        for stop_guid in self.manager_stop:
            dict_idx[comp] = stop_guid
            comp +=1

        return dict_idx

    def create_data_model(self):
        """Stores the data for the problem."""
        data = dict()
        dist_matrix = [[0.0 for i in self.dict_idx_guid] for j in self.dict_idx_guid ]
        for i in self.dict_idx_guid:
            if i == 0:
                stop_chosen = self.manager_stop.depot
            else:
                stop_chosen = self.manager_stop[self.dict_idx_guid[i]]
            for j in self.dict_idx_guid:
                if j == 0:
                    stop_tried = self.manager_stop.depot
                else:
                    stop_tried = self.manager_stop[self.dict_idx_guid[j]]
                dist_matrix[i][j] = self.multiplier * stop_chosen.get_dist_another_stop(stop_tried)        # we multiplied by 1000 because
                                                                                                # google or takes integer as input

        data['distance_matrix'] = dist_matrix

        list_demand = [0] + [self.manager_stop[self.dict_idx_guid[i]].demand for i in self.dict_idx_guid if i != 0]
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


        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager_model.IndexToNode(from_index)
            to_node = manager_model.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


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

        if self.min_veh:
            penalty_veh = self.multiplier * 1000
            for i in range(data['num_vehicles']):
                routing.SetFixedCostOfVehicle(penalty_veh, i)

        # Allow to drop nodes.
        penalty = self.multiplier * 100000
        for node in range(1, len(data['distance_matrix'])):
            routing.AddDisjunction([manager_model.NodeToIndex(node)], penalty)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.time_limit.seconds = 10

        # Solve the problem.
        assignment = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if assignment:
            return self._parse_solution(data, manager_model,routing, assignment)


    def _parse_solution(self,data,manager_model,routing, assignment):
        """
        Parse solution on a form interpretable
        :return: a sequence of stops, including depot
        """
        seq_stop = [self.manager_stop.depot]
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)

            nb_stop = 0
            while not routing.IsEnd(index):
                node_index = manager_model.IndexToNode(index)
                if nb_stop >0:
                    stop_guid = self.dict_idx_guid[node_index]
                    seq_stop.append(self.manager_stop[stop_guid])
                index = assignment.Value(routing.NextVar(index))
                nb_stop +=1

            # add depot at the end if vehicles
            if nb_stop >1:
                seq_stop.append(self.manager_stop.depot)

        return seq_stop

