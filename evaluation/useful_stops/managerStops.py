
from evaluation.useful_stops import stop

class managerStops(dict):
    """
    Gather all stop under a dict form
    """

    def __init__(self):
        super(managerStops,self).__init__()
        self.depot = None
    
    @property
    def demand(self):
        return sum(st.demand for st in self.values())

    @classmethod
    def from_line(cls,line):
        """
        Create a manager from a line of a file
        :param line: the line of a file
        :return: a managerStops object
        """
        words = line.strip().split(" ")
        nb_stops = int(len(words)/3)
        mana = cls()

        for i in range(0,nb_stops-1):
            new_guid = mana._create_guid()
            mana[new_guid] = stop.Stop(guid=new_guid,
                                        x= words[3*i],
                                       y= words[3*i+1],
                                       demand=words[3*i+2])
        # create depot
        depot =stop.Stop(guid="depot",x=words[-3],
                               y = words[-2],demand=words[-1])
        mana._set_depot(depot)
        return mana

    @classmethod
    def from_stops_list(cls,manager_ref,list_stops):
        """
        From a mangaer ref, creates a manager as a subset composed of the stops within the list
        :param manager_ref: the initial manager containing all the stops
        :param list_stops: the list of stops we are interesting in
        :return: a new manager
        """
        mana = cls()
        mana._set_depot(manager_ref.depot)

        for stop_considered in list_stops:
            mana[stop_considered.guid] = stop_considered

        return mana

    def _create_stop_from_stop(self,stop_copied, demand):
        """
        Create a new stop from the one to be copied, except for the demand
        :param stop_copied:
        :param demand:
        :return:
        """
        new_guid = self._create_guid()
        self[new_guid] = stop.Stop(guid=new_guid,
                                   x=stop_copied.x,
                                   y=stop_copied.y,
                                   demand=demand)

    def _set_depot(self,depot):
        self.depot = depot


    def _create_guid(self):
        guid = "stop_" + str(len(self))
        assert not guid in self
        return guid


    def check_capacity(self,cap):
        """
        Check that every stop fit the cap
        :param cap: the max cap of the truck
        :return: update itself
        """
        initial_demand = self.demand
        list_stop_to_deal = [st.guid for st in self.values() if st.demand > cap]

        for stop_id in list_stop_to_deal:
            stop_considered = self[stop_id]
            q,r = divmod(stop_considered.demand,cap)
            stop_considered.demand = r
            for i in range(0,q):
                self._create_stop_from_stop(stop_considered,cap)

        assert initial_demand == self.demand, str(initial_demand) + "_" + str(self.demand)






