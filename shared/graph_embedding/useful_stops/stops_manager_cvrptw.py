from shared.graph_embedding.useful_stops import stops_manager_cvrp
from shared.graph_embedding.useful_stops import stops_cvrptw
from shared.graph_embedding.useful_stops import classDepot

import numpy as np

class StopsManagerCVRPTW(stops_manager_cvrp.StopsManagerCRVP):
    """
    Class of manager for stops_cvrptw
    """

    def __init__(self,depot = None):
        super(StopsManagerCVRPTW,self).__init__(depot)


    @classmethod
    def from_cvrpTW_file(cls,filename):
        """
        Create a stop manager filled
        :param filename: the file from which we should read the stops
        :return: an object of this class
        """
        manager = cls()
        file = open(filename, 'r')  # reading only
        line = file.readline()
        reached_customer_section = False
        while line:
            words = line.strip().split("\t")

            # check if the line corresponds to a stop
            if reached_customer_section:
                manager._create_stop(line)

            # check if next one is going to be
            else:
                if len(words) >= 1 and words[0] == "CUSTOMER":
                    reached_customer_section = True
                    # we have to pass the line corresponding to the headers and the blank one
                    file.readline()
                    file.readline()
                    line = file.readline()
                    manager._create_depot(line)

            line = file.readline()

        file.close()

        if not reached_customer_section:
            assert False

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
        manager.depot = classDepot.Depot(words[0],words[1],words[3])

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
                    manager[guid] = stops_cvrptw.Stop_cvrptw(guid,st_w[0],st_w[1],st_w[4],st_w[2],st_w[3],0)

        return manager


    def _create_stop(self,line):
        """
        From the line of the file, create a stop with the corresponding
        :param line: a line from the file
        :return:
        """
        words = line.strip().split(" ")

        try:
            words = [int(wo) for wo in words if wo != '']
        except:
            words = line.strip().split("\t")
            if len(words) != 7:
                assert False



        guid = self._check_guid(words[0])
        self[guid] = stops_cvrptw.Stop_cvrptw(guid, words[1], words[2], words[3], words[4], words[5], words[6])

    def _create_depot(self,line):
        """
        From the line of the file create the corresponding depot
        :param line: a line from the input
        :return:
        """
        words = line.strip().split(" ")
        words = [wo for wo in words if wo != '']
        if len(words) != 7:
            words = line.strip().split("\t")
            if len(words) != 7:
                assert False
        assert int(words[0]) == 0, words

        self.depot = classDepot.Depot(words[1], words[2], words[5])


    def dump_to_file(self,file):
        """
        Dump the manager stop in the right format in the file precised over there
        :param file: the output file
        :return: a dict matching newId to oldId
        """
        file.write("CUSTOMER\n")
        file.write("CUST NO.\tXCOORD.\tYCOORD.\tDEMAND\tREADY TIME\tDUE DATE\tSERVICE TIME\n")
        file.write("\n")
        self._dump_depot_section(file)
        dict_new_old_id = self._dump_node_section(file)
        return dict_new_old_id

    def _dump_depot_section(self,file):
        """
        Writhe the depot line in the corresponding line
        :param file:
        :return:
        """
        depot_txt = "0 \t " + str(self.depot.x) + "\t" + str(self.depot.y) + "\t 0 \t 0 \t" + str(self.depot.due_date) + "\t 0"
        file.write(depot_txt +"\n")

    def _dump_node_section(self,file):
        """
        Dump all the nodes to the right files
        :param file:
        :return: a dict matching file id to stop ID
        """
        comp = 0
        dict_new_old = {}
        for stopId in self.keys():
            comp +=1
            stop = self[stopId]
            text_stop = str(comp) + "\t" + str(stop.x) + "\t" + str(stop.y) + "\t" + str(stop.demand) + "\t" + str(stop.beginTW) +\
                "\t" + str(stop.endTW) + "\t" + str(stop.service_time)
            file.write(text_stop + "\n")
            dict_new_old[comp] = stopId

        return dict_new_old


    def get_all_stops_end_in(self,start_time,end_time):
        """
        Go through all stops whose end tw ends in [start_time, end_time[
        :param start_time: begining of interval
        :param end_time: end of interval
        :return: a list of stop Ids
        """
        list_stop_id = []
        for stopId in self.keys():
            stop = self[stopId]
            if start_time<= stop.endTW< end_time:
                list_stop_id.append(stopId)

        return list_stop_id

    def get_list_overlapping_stop(self,stop,threshold):
        """
        Go through all the stops determine the number of overlapping stop (i.e sharing at least the theshold)
         of the tw
         :param stop: the considered stop
         :param threshold: the amount of time the tw has to overlapp to count it as overlapping
        :return: the list of stop id sharing the tw
        """
        list_overlap = []
        for otherId in self.keys():
            other = self[otherId]

            if stop.endTW - other.beginTW >= threshold and other.endTW - stop.beginTW >= threshold:
                list_overlap.append(otherId)
        return list_overlap


    def get_avg_number_overlapping_tw_stops(self,threshold):
        """
        Go through all the stops, and for each one, determine the number of overlapping stop (i.e sharing at least the theshold)
         of the tw
         :param threshold: the amount of time the tw has to overlapp to count it as overlapping
        :return: the avergage number of such stops
        """
        list_overlap = []
        for stopId in self.keys():
            stop = self[stopId]
            list_overlap.append(len(self.get_list_overlapping_stop(stop,threshold)))

        return np.mean(list_overlap)







