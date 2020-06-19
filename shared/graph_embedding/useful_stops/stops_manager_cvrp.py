
from shared.graph_embedding.useful_stops import stops_cvrp, classDepot,stops_manager


class StopsManagerCRVP(stops_manager.StopsManager):
    """
    Inherits from dict, gather all orders at the same place
    """

    def __init__(self,depot = None):
        super(StopsManagerCRVP, self).__init__(depot)


    @classmethod
    def from_cvrp_file(cls,filename):
        """
        Create a stop manager filled
        :param filename: the file from which we should read the stops
        :return: an object of this class
        """
        manager = cls()
        file = open(filename, 'r')  # reading only
        line = file.readline()
        reached_coord_section = False
        reached_demand_section = False
        reached_depot_section = False
        while line:
            words = line.strip().split(" ")
            # check if the line corresponds to the demand
            if reached_demand_section:
                if words[0] == 'DEPOT_SECTION':
                    reached_demand_section = False
                    reached_depot_section = True
                else:
                    manager._update_demand(line)
            # check if it corresponds to the coordinate
            elif reached_coord_section:
                if words[0] == 'DEMAND_SECTION':
                    reached_demand_section = True
                    reached_coord_section = False
                else:
                    manager._create_stop(line)
            # check if corresponds to depot section
            elif reached_depot_section:
                manager._create_depot(line)
                reached_depot_section = False
            # check if next one is going to be
            else:
                if words[0] == "NODE_COORD_SECTION":
                    reached_coord_section = True
                elif words[0] == 'DEMAND_SECTION':
                    reached_demand_section = True

            line = file.readline()

        file.close()
        return manager


    @classmethod
    def from_txt_transfer(cls,line):
        """
        Create a stop manager from the transfer text input
        :param line: a line of the input, corresponding to multiple stops
        """
        manager = cls()

        # create the depot
        depot_section = line.split('*')[1]
        words = depot_section.split('-')
        manager.depot = classDepot.Depot(words[0],words[1])

        # create the stops
        stops_section = line.split('*')[2]
        words = stops_section.split('_')
        for i,w in enumerate(words):
            if len(w) == 0:
                assert i == len(words) -1, print(i,w,len(words))
            else:
                st_w = w.split('-')
                if len(st_w) == 1:
                    assert st_w[0] == '\n', print(st_w)
                else:
                    guid = manager.get_guid(i)
                    if 'e' in w:
                        # find the indice i
                        for i,nu in enumerate(st_w):
                            if 'e' in nu:
                                break
                        else:
                            assert False
                        st_w[i] = 0
                        st_w.pop(i+1)
                    manager[guid] = stops_cvrp.Stop_cvrp(guid,st_w[0],st_w[1],st_w[2])

        return manager


    @classmethod
    def from_array(cls, input_array):
        """
        From the array, build a manager stop
        :param input_array: an array [number_stops x dim]
        :return: a manager stop
        """
        manager = cls()

        # create the depot
        depot_section = input_array[-1,:]
        manager.depot = classDepot.Depot(depot_section[0],depot_section[1])


        nb_stops = len(input_array[:,0]) -1
        for i in range(0, nb_stops):
            guid = manager.get_guid(i)
            manager[guid] = stops_cvrp.Stop_cvrp(guid,input_array[i,0],input_array[i,1],input_array[i,2])

        return manager


    def _create_stop(self,line):
        """
        From the line of the file, create a stop with the corresponding
        :param line: a line from the file
        :return: a stop object
        """
        words = line.strip().split(" ")
        if len(words) != 3:
            assert False

        guid = self._check_guid(words[0])
        self[guid] = stops_cvrp.Stop_cvrp(guid, words[1], words[2], 0)

    def _create_depot(self,line):
        """
        Create a depot
        :param line: the line correspondoing to the depot point
        :return:
        """
        words = line.strip().split(" ")
        if len(words) != 2:
            assert False
        self.depot = classDepot.Depot(words[0], words[1])


    def _update_demand(self,line):
        """
        From the line of the file set the demand of the corresponding stop
        :param line: line from the file
        :return:
        """
        words = line.strip().split(" ")
        if len(words) != 2:
            assert False

        guid = self.get_guid(words[0])
        self[guid].demand = int(words[1])


    def check_demand_updated(self):
        """
        Check that the demand is not null for any of the stops
        :return: a boolean
        """
        for stop_id in self.keys():
            if self[stop_id].demand == 0 and self[stop_id].stop_type ==2:
                return False

        return True

    def dump_to_file(self,file):
        """
        Write in the vrp format the manager stop
        :param file: the corresponding file
        :return: a map[id in file] = stopId
        """
        dimension_text = "DIMENSION : " + str(len(self) +1)
        file.write(dimension_text + "\n")
        map_new_old_id = self._dump_node_coord_section(file)
        self._dump_node_demand_section(file,map_new_old_id)
        self._dump_depot_section(file)
        return map_new_old_id


    def _dump_node_demand_section(self,file,map_new_old_id):
        """
        Write in the vrp format the demand section
        :param file: the corresponding file
        :param map_new_old_id: a dict[new_stop_id] = old_stop_id
        :return:
        """
        demand_text = "DEMAND_SECTION"
        file.write(demand_text + "\n")
        # First corresponds to the depot, with a demand of zero
        text_demand = str(1) + " " + str(0)
        file.write(text_demand +"\n")
        for newId in map_new_old_id.keys():
            stopId = map_new_old_id[newId]
            stop = self[stopId]
            text_demand = str(newId) + " " + str(stop.demand)
            file.write(text_demand +"\n")

    @property
    def demand(self):
        return sum(stop.demand for stop in self.values())










