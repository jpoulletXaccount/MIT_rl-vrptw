import numpy as np

class Route(object):
    """
    Class of a route, i.e. a succession of stops with return to the depot
    """

    def __init__(self,sequence_stop,guid = None):
        self.sequence_stops = sequence_stop
        self.guid = guid


    @property
    def nb_stops(self):
        """
        :return: the true number of stops (no depot)
        """
        depot = self.sequence_stops[0]
        nb_true_stop = 0
        for stop in self.sequence_stops:
            if abs(depot.x - stop.x) > 0.0001 or abs(depot.y - stop.y) > 0.0001:
               nb_true_stop +=1

        return nb_true_stop

    @property
    def length(self):
        """
        :return: the total length of a route
        """
        tot_dist = 0

        for i in range(0,self.nb_stops-1):
            prev_stop = self.sequence_stops[i]
            current_stop = self.sequence_stops[i+1]

            tot_dist += np.sqrt((prev_stop.x - current_stop.x)**2 + (prev_stop.y - current_stop.y)**2)

        return tot_dist

    @property
    def time(self):
        """
        Note, only makes sense to be called if we are dealing with tw
        :return: the total time needed for the route
        """
        tot_time = 0
        current_time = 0

        depot = self.sequence_stops[0]

        for i in range(0,self.nb_stops-1):
            prev_stop = self.sequence_stops[i]
            current_stop = self.sequence_stops[i+1]

            time_traveled = np.sqrt((prev_stop.x - current_stop.x)**2 + (prev_stop.y - current_stop.y)**2)
            current_time = max(current_time + time_traveled, current_stop.begin_tw)

            # check if is depot or not
            if abs(depot.x - current_stop.x) <= 0.0001 and abs(depot.y - current_stop.y) <= 0.0001:
                tot_time += current_time
                current_time = 0

        return tot_time

    
    @property
    def demand(self):
        """
        :return: the total demand of the route
        """
        return sum(stop.demand for stop in self.sequence_stops)

    @property
    def vehicles(self):
        """
        :return: the number of time the vehicles return to depot
        """
        depot = self.sequence_stops[0]
        nb_visit_depot = 0
        for stop in self.sequence_stops:
            if abs(depot.x - stop.x) <= 0.0001 and abs(depot.y - stop.y) <= 0.0001:
                nb_visit_depot +=1

        assert nb_visit_depot >=2, nb_visit_depot

        return nb_visit_depot -1

    @property
    def first_stop(self):
        return self.sequence_stops[1]

    @property
    def last_stop(self):
        return self.sequence_stops[-2]

    @property
    def depot(self):
        return self.sequence_stops[0]

    def get_only_true_stops(self):
        """
        :return: the list of all stops within the route
        """
        depot = self.sequence_stops[0]
        list_true_stop = []
        for stop in self.sequence_stops:
            if stop.demand >= 1:
                assert abs(depot.x - stop.x) > 0.0001 or abs(depot.y - stop.y) > 0.0001, stop.to_print()
                list_true_stop.append(stop)

        return list_true_stop


    def merge_with_another_route(self,other_route):
        """
        Merge the current route with the other route, in this sense
        :param other_route: a route object
        """
        inital_nb_stop = self.nb_stops + other_route.nb_stops
        new_sequence = self.sequence_stops[0:-1]
        new_sequence.extend(other_route.sequence_stops[1:])

        self.sequence_stops = new_sequence

        assert inital_nb_stop == self.nb_stops

    def insert_stop(self,insertion_pos, stop):
        """
        Insert a stop just after the insertion pos
        :param insertion_pos: eg if 0 then will be inserted just after the depot
        :param stop: the stop considered
        :return:
        """
        new_sequence = self.sequence_stops[:insertion_pos+1] + [stop] + self.sequence_stops[insertion_pos+1:]
        self.sequence_stops = new_sequence


    def check_feasible_tw(self):
        """
        Check that the route is feasible with respect to tw
        :return: a boolean
        """
        current_time = 0
        list_time = [current_time]
        depot = self.sequence_stops[0]

        for i in range(0,self.nb_stops-1):
            prev_stop = self.sequence_stops[i]
            current_stop = self.sequence_stops[i+1]

            time_traveled = np.sqrt((prev_stop.x - current_stop.x)**2 + (prev_stop.y - current_stop.y)**2)
            current_time = max(current_time + time_traveled, current_stop.begin_tw)


            if current_time > current_stop.end_tw:
                print("This stop is not valid ", current_time, current_stop.end_tw)
                print([stop.to_print() for stop in self.sequence_stops])
                print(list_time)
                return False

            # check if is depot or not
            if abs(depot.x - current_stop.x) <= 0.000001 and abs(depot.y - current_stop.y) <= 0.000001:
                current_time = 0

            list_time.append(current_time)

        return True



