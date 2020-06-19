import numpy as np

from evaluation.useful_stops import route

class I1heuristics(object):
    """
    Implement the I1 heuristics
    """

    def __init__(self,manager_stop,env):
        self.manager_stop = manager_stop
        self.env= env
        self.matrix_dist = self._build_dist_matrix()
        self.current_route_dict = dict()
        self.current_route = None

        self.un_assigned_stop = list(self.manager_stop.keys())

        # Param
        self.ALPHA_1 = 0.5
        self.ALPHA_2 = 0.5
        assert self.ALPHA_1 + self.ALPHA_2 ==1

        self.MU = 1
        self.LAMBDA = 1

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
        Create one route (i.e one vehicle) with the stop the furthest away from the depot
        :return: a dict[route_id] = route
        """
        depot_id = self.manager_stop.depot.guid
        list_dist_depot = [(self.matrix_dist[depot_id][stop_id],stop_id) for stop_id in self.un_assigned_stop]
        list_dist_depot.sort(reverse=True)

        further = list_dist_depot[0][1]
        stop = self.manager_stop[further]
        new_guid = "route_" + str(len(self.current_route_dict))
        short_sequence = [self.manager_stop.depot,stop, self.manager_stop.depot]
        self.current_route_dict[new_guid] = route.Route(sequence_stop=short_sequence,guid=new_guid)
        self.current_route = self.current_route_dict[new_guid]

        self.un_assigned_stop.remove(further)


    def _compute_c11(self,pos_i,stop):
        """
        Compute the c11 cost of inserting stop after pos i in the current route
        :param pos_i:
        :return: a cost
        """
        stop_i = self.current_route.sequence_stops[pos_i]
        stop_j = self.current_route.sequence_stops[pos_i + 1]

        c11 = self.matrix_dist[stop_i.guid][stop.guid] + self.matrix_dist[stop.guid][stop_j.guid] - self.MU * self.matrix_dist[stop_i.guid][stop_j.guid]

        return c11


    def _get_begin_ser(self,seq_stop):
        """
        Return the time at which we arrive at the last stop
        :param seq_stop: a sequence of stop objct
        :return: a time
        """
        current_time = 0

        for i in range(0,len(seq_stop)-1):
            prev_stop = seq_stop[i]
            current_stop = seq_stop[i+1]

            time_traveled = self.matrix_dist[prev_stop.guid][current_stop.guid]
            current_time = max(current_time + time_traveled, current_stop.begin_tw)

        return current_time


    def _compute_c12(self,pos_i,stop):
        """
        Compute the c12 cost of inserting stop after pos i in the current route
        :param pos_i:
        :param stop:
        :return:
        """
        stop_j = self.current_route.sequence_stops[pos_i + 1]
        bj = self._get_begin_ser(seq_stop=self.current_route.sequence_stops[:(pos_i + 2)])
        with_stop  = self.current_route.sequence_stops[:(pos_i +1)] + [stop,stop_j]
        bju = self._get_begin_ser(seq_stop=with_stop)

        return bju - bj

    def _computec1(self,pos_i,stop):
        """
        Compute the c1 cost of inserting stop after pos i in the current route
        :param pos_i:
        :param stop:
        :return: the score c1
        """
        c1 = self.ALPHA_1 * self._compute_c11(pos_i,stop) + self.ALPHA_2 * self._compute_c12(pos_i, stop)

        return c1

    def _computec2(self,stop,c1):
        """
        Compute the score c2
        :param stop:
        :param c1:
        :return: the score c2
        """
        depot_id = self.manager_stop.depot.guid
        c2 = self.LAMBDA * self.matrix_dist[depot_id][stop.guid] - c1
        return c2


    def _check_feas_sequence(self,seq_stop):
        """
        Check if the sequence is feasible or not
        :param seq_stop: a sequence of stop objct
        :return: a boolean
        """
        # Check cap
        tot_cap = sum([stop.demand for stop in seq_stop])
        if tot_cap > self.env.capacity:
            return False

        current_time = 0
        depot = seq_stop[0]
        for i in range(0,len(seq_stop)-1):
            prev_stop = seq_stop[i]
            current_stop = seq_stop[i+1]

            time_traveled = self.matrix_dist[prev_stop.guid][current_stop.guid]
            current_time = max(current_time + time_traveled, current_stop.begin_tw)

            if current_time > current_stop.end_tw:
                return False

            # check if is depot or not
            if abs(depot.x - current_stop.x) <= 0.0001 and abs(depot.y - current_stop.y) <= 0.0001:
                assert i+1 == len(seq_stop) -1

        return True


    def _check_feas_insertion(self,pos_i,stop):
        """
        Check if we can insert the stop at the pos i in the actual route
        :param pos_i: the position of i
        :param stop: the stop considereds
        :return: a boolean
        """
        seq_to_try = self.current_route.sequence_stops[:(pos_i+1)] + [stop] + self.current_route.sequence_stops[(pos_i+1):]

        return self._check_feas_sequence(seq_stop=seq_to_try)


    def _concatenate_all_routes(self):
        """
        Merge all routes left
        :return: a route object
        """
        final_sequence_stop = []
        for route_id in self.current_route_dict:
            route_considered = self.current_route_dict[route_id]

            final_sequence_stop.extend(route_considered.sequence_stops[0:-1])  # we take everything but the last depot

        # need to add the final depot
        final_sequence_stop.append(self.manager_stop.depot)

        final_route = route.Route(sequence_stop=final_sequence_stop,guid="final_route")

        assert final_route.nb_stops == len(self.manager_stop),print(final_route.nb_stops,len(self.manager_stop))
        return final_route


    def solve(self):
        """
        Solve the routing problem performing the I1 heuristics
        :return: a route, with multiples return to the depot if needed
        """
        self._initialize_routes()

        while len(self.un_assigned_stop) >0:

            best_stop_to_insert = []        # keep track of the stop to insert and the position

            for stopid in self.un_assigned_stop:
                stop_considered = self.manager_stop[stopid]

                feasible_inser = [i for i in range(0,len(self.current_route.sequence_stops) -1) if self._check_feas_insertion(i,stop_considered)]
                list_c1 = [(self._computec1(i,stop_considered),i) for i in feasible_inser]

                if len(list_c1) >0:
                    list_c1.sort()
                    best_position = list_c1[0][1]

                    best_stop_to_insert.append((self._computec2(stop_considered,list_c1[0][0]),best_position,stop_considered))

            # check if we have found feasible stops to insert
            if len(best_stop_to_insert) >0:
                best_stop_to_insert.sort(reverse=True)
                pos_to_insert = best_stop_to_insert[0][1]
                stop_to_insert = best_stop_to_insert[0][2]

                self.current_route.insert_stop(pos_to_insert,stop_to_insert)
                self.un_assigned_stop.remove(stop_to_insert.guid)

            else:
                self._initialize_routes()

        # We finally merge all the route to obtain the "big" final route
        final_route = self._concatenate_all_routes()

        return final_route.sequence_stops




