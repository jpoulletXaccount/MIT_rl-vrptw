
import numpy as np

class StopsManager(dict):
    """
    Inherits from dict, gather all orders at the same place
    """
    def __init__(self,depot=None):
        super(StopsManager, self).__init__()
        self.depot = depot  # corresponds to the depot


    def _create_guid(self):
        """
        Create a unique Id per stop
        :return: the Id
        """
        test_id = "stop_" + str(len(self) +1)
        if test_id in self.keys():
            assert False

        return test_id

    def _check_guid(self,int_id):
        """
        Check if the guid derived from the number if available or not
        :param int_id: the number
        :return: the id
        """
        test_id = "stop_" + str(int_id)
        if test_id in self.keys():
            assert False

        return test_id

    @staticmethod
    def get_guid(int_id):
        """
        Return the id deriving from the int_id
        :param int_id: a number
        :return: string id
        """
        return "stop_" + str(int_id)

    def _dump_node_coord_section(self,file):
        """
        Write in a vrp format the coodinate of the node
        :param file: the corresponding file
        :return: a map[newId] = old_Id
        """
        map_new_old_id = {}
        begin_section_text = 'NODE_COORD_SECTION'
        file.write(begin_section_text + "\n")
        # First node corresponds to the depot
        text_node = str(1) + " " + str(self.depot.x) + " " + str(self.depot.y)
        file.write(text_node +"\n")
        comp = 1
        for stopId in self.keys():
            comp +=1
            stop = self[stopId]
            text_node = str(comp) + " " + str(stop.x) + " " + str(stop.y)
            file.write(text_node +"\n")
            map_new_old_id[comp] = stopId

        return map_new_old_id

    def _dump_depot_section(self,file):
        """
        Write the depot section in the appropriate format in the file
        :param file: the output file
        :return:
        """
        depot_section_text = "DEPOT_SECTION"
        assert not self.depot is None
        file.write(depot_section_text + "\n")
        file.write("1 \n")
        file.write("-1 \n")
        file.write("EOF")


    @classmethod
    def init_from_cluster(cls,cluster):
        """
        Init a manager stop from a cluster object
        :param cluster: a cluster object
        :return: a manager stop
        """
        current_manager = cls()
        for stop in cluster.list_stops:
            assert stop.guid not in current_manager.keys()
            current_manager[stop.guid] = stop

        return current_manager


    def set_depot(self,depot):
        """
        Set the depot value
        :param depot: a depot object
        :return:
        """
        self.depot = depot


    def get_diameter(self):
        """
        :return: the diameter = the max pairwise distance
        """
        if len(self) <=1:
            return 0

        tot_pairwise_dist = []
        for stop_1 in self.values():
            for stop_2 in self.values():
                dist = stop_1.get_distance_to_another_stop(stop_2)
                tot_pairwise_dist.append(dist)

        return max(tot_pairwise_dist)


    @classmethod
    def from_sublist(cls,list_stops,reference_manager_stop):
        """
        Create the manager stops corresponding to all stops listed in sublist
        :param list_stops: the list of stops to be put in the manager stops
        :param reference_manager_stop: the overall manager stops
        :return: an object manager stops
        """
        new_manager = cls(depot=reference_manager_stop.depot)
        for stopId in list_stops:
            if "stop_" in stopId:
                new_stopID = stopId
            else:
                new_stopID = "stop_" + str(stopId)
            new_manager[new_stopID] = reference_manager_stop[new_stopID]

        return new_manager


    @property
    def centroid(self):
        x_mean = np.mean([stop.x for stop in self.values()])
        y_mean = np.mean([stop.y for stop in self.values()])
        return x_mean,y_mean






