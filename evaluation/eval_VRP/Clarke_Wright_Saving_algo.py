
from evaluation.useful_stops import route

class ClarkeWrightSolver(object):
    """
    Implement the Clarke Wright saving algorithm to solve the routing problem
    """

    def __init__(self,manager_stop,env):
        self.manager_stop = manager_stop
        self.env= env
        self.matrix_dist = self._build_dist_matrix()
        self.current_route = dict()


    def _build_dist_matrix(self):
        """
        Build the distance matrix between every stop and the depot
        :return: a dict[stop_guid][stop_guid] = dist
        """
        final_matrix = {}
        for stop_guid_1 in self.manager_stop:
            final_matrix[stop_guid_1] = {}
            stop1 = self.manager_stop[stop_guid_1]
            for stop_guid_2 in self.manager_stop:
                stop2 = self.manager_stop[stop_guid_2]
                final_matrix[stop_guid_1][stop_guid_2] = stop1.get_dist_another_stop(stop2)

        # Add depot
        final_matrix[self.manager_stop.depot.guid] = {}
        for stop_guid in self.manager_stop:
            stop = self.manager_stop[stop_guid]

            final_matrix[stop_guid][self.manager_stop.depot.guid] = stop.get_dist_another_stop(self.manager_stop.depot)
            final_matrix[self.manager_stop.depot.guid][stop_guid] = self.manager_stop.depot.get_dist_another_stop(stop)

        return final_matrix


    def _initialize_routes(self):
        """
        Create one route (i.e one vehicle) for each stop
        :return: a dict[route_id] = route
        """

        for i,stop_id in enumerate(list(self.manager_stop.keys())):
            stop = self.manager_stop[stop_id]

            assert stop.demand <= self.env.capacity, stop.demand

            new_guid = "route_" + str(i)
            short_sequence = [self.manager_stop.depot,stop, self.manager_stop.depot]
            self.current_route[new_guid] = route.Route(sequence_stop=short_sequence,guid=new_guid)


    def _compute_savings(self):
        """
        Compute the savings of merging two routes
        :return: a list of (saving, route1_guid + "-" + route2_guid
        """
        list_savings = []
        for route1_guid in self.current_route:
            route1 = self.current_route[route1_guid]
            stop1 = route1.last_stop

            for route2_guid in self.current_route:
                if route2_guid != route1_guid:
                    route2 = self.current_route[route2_guid]
                    # check feasibility
                    if route1.demand + route2.demand <= self.env.capacity:
                        stop2 = route2.first_stop

                        saving = self.matrix_dist[stop1.guid][self.manager_stop.depot.guid] + self.matrix_dist[self.manager_stop.depot.guid][stop2.guid] +\
                            self.matrix_dist[stop1.guid][stop2.guid]

                        id_couple = route1_guid + "-" + route2_guid
                        list_savings.append((saving, id_couple))

        return list_savings


    def _get_route_from_tuple(self,couple):
        """
        :param couple:  route1_guid + "-" + route2_guid
        :return: route1, route2
        """
        words = couple.split("-")
        route1_guid = words[0]
        route_1 = self.current_route[route1_guid]
        route2_guid = words[1]
        route_2 = self.current_route[route2_guid]

        return route_1,route_2

    def _concatenate_all_routes(self):
        """
        Merge all routes left
        :return: a route object
        """
        final_sequence_stop = []
        for route_id in self.current_route:
            route_considered = self.current_route[route_id]

            final_sequence_stop.extend(route_considered.sequence_stops[0:-1])  # we take everything but the last depot

        # need to add the final depot
        final_sequence_stop.append(self.manager_stop.depot)

        final_route = route.Route(sequence_stop=final_sequence_stop,guid="final_route")

        assert final_route.nb_stops == len(self.manager_stop)
        return final_route


    def solve(self):
        """
        Solve the routing problem following Clarke-Wright saving heuristics
        :return: a route (with multiples return to the depot)
        """
        self._initialize_routes()

        changes = True
        while changes:

            current_list_savings = self._compute_savings()
            current_list_savings.sort(reverse=True)

            for tuple_saving in current_list_savings:
                route_1,route_2 = self._get_route_from_tuple(tuple_saving[1])

                route_1.merge_with_another_route(route_2)
                del self.current_route[route_2.guid]
                break

            else:
                # we have not break thus no changes
                changes = False


        # We finally merge all the route to obtain the "big" final route
        final_route = self._concatenate_all_routes()

        return final_route.sequence_stops


