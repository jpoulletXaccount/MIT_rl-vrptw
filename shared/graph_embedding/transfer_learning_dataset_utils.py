
import numpy as np
import time
from tqdm.auto import tqdm

from sklearn.preprocessing import normalize
from shared.graph_embedding.useful_files.sparse_graph_task import DataFold
from shared.graph_embedding.useful_stops import stops_manager_cvrp,stops_manager_cvrptw

class TransferDatasetUtils(object):
    """
    Class useful to parse the data corresponding to the created data set
    """

    def __init__(self,path, number_type,true_dist):
        # if isinstance(path,str):
        #     path = path + "/vrp_transfer.txt"
        #     self.list_data = self._parse_txt_input(path)
        #
        # else:
        #     self.list_data = None

            # self.list_data = self._parse_rl_input(path)

        # to be filled later on
        self.list_dist_matrix = []
        self.list_type_num = []
        self.list_features = []
        self.list_labels = []

        # Parameters
        self._true_dist = true_dist
        self.MAX_VEHI = None
        self._number_features = 3
        self._number_type = number_type     # Determine the number of type of link
        self._scale = [5,12,25,50,100]
        self._scale = [i * np.sqrt(2)/100 for i in self._scale]     # rescale to the square

        # self._scale = [5,10,15,25,50,75,100,150,200,500]
        assert len(self._scale) == self._number_type or true_dist


    @property
    def number_features(self):
        return self._number_features

    @property
    def number_labels(self):
        return self.MAX_VEHI


    def fast_load_data_path(self,path):
        list_label, input_path = self._parse_txt_input_to_array(path)
        self.MAX_VEHI = max(list_label)

        self.list_dist_matrix, self.list_type_num, self.list_features, self.list_labels = self.fast_load_data_test(input_path)

        return self.list_dist_matrix, self.list_type_num, self.list_features,list_label

    def fast_load_data_test(self,input_data):
        """
        Try to speed up the loading of the data
        :param input_data: a numpy array given by the data generator
        :return: everything ready for the DataFold.TEST
        """
        batch_size, num_node, num_features = input_data.shape

        if num_features == 5:
            # rearrange the features
            permutation = [0,1,4,2,3]
            input_data_features = input_data[:,:,permutation]

            # add the column
            col_service_time = np.zeros(shape=(batch_size,num_node,1))
            input_data_features = np.concatenate((input_data_features,col_service_time),axis=2)
            num_features +=1    # (since we have added the service time)

        else:
            input_data_features = input_data

        self._number_features = num_features
        self.list_labels = [0 for _ in range(batch_size)]

        input_concat = np.concatenate(input_data_features)
        norm_by_feature = np.reshape(np.transpose(input_concat),(num_features,-1))
        norm_by_feature = normalize(norm_by_feature, axis=1)
        self.list_features = np.reshape(np.transpose(norm_by_feature),(batch_size,num_node,num_features))

        for graph in range(0,batch_size):
            edge_matrix = []
            matrix_type_num = np.zeros(shape=(self._number_type,num_node))
            #  Create dist matrix
            considered_stops = input_data[graph,:,:2]
            square_input = np.sum(np.square(considered_stops),axis=1)
            row = np.reshape(square_input, (-1,1))
            col= np.reshape(square_input,(1,-1))
            dist_mat = np.sqrt(np.maximum(row -2 * np.matmul(considered_stops,np.transpose(considered_stops)) + col,0.0))
            not_masked = np.ones_like(dist_mat)
            np.fill_diagonal(not_masked,0)  # don't consider your self
            for i,max_dist in enumerate(self._scale):
                true_for_edge = np.less_equal(dist_mat,max_dist)
                true_for_edge = np.logical_and(not_masked,true_for_edge)

                # get all the links which belong to the scale considered
                indices_i, indices_j = np.where(true_for_edge)
                indices = np.stack([indices_i,indices_j])
                indices = np.reshape(np.transpose(indices), (-1,2))
                dist_spe = dist_mat[indices_i,indices_j]
                indices = np.concatenate([indices,np.reshape(dist_spe,(-1,1))],axis=1)

                # update the mask
                not_masked = np.logical_and(not_masked,np.logical_not(true_for_edge))

                # compute interesting data
                edge_matrix.append(indices)
                matrix_type_num[i,:] = np.sum(true_for_edge,0)

            self.list_dist_matrix.append(edge_matrix)
            self.list_type_num.append(matrix_type_num)

        return self.list_dist_matrix, self.list_type_num, self.list_features, self.list_labels



    def load_data_test(self):
        """
        Load the data and process them
        :return: everything ready for the DataFold.TEST
        """
        for t in tqdm(self.list_data):
            self.parse_tuple(t)

        self.normalize_inverse_distance()
        self.normalize_features()

        return self.list_dist_matrix, self.list_type_num, self.list_features, self.list_labels

    def load_data(self):
        """
        Main function, load the data in self.df_data
        :return: a dict[train] = list dist matrix, a dict[train] = list_feature, a dict[train] = list label
        """
        for t in tqdm(self.list_data):
            self.parse_tuple(t)

        self.normalize_inverse_distance()
        self.normalize_features()

        return self.split_data()

    @staticmethod
    def _parse_rl_input(input):
        """
        Parse the input give by the rl agent
        :param input: an array [batch_size x max_time x dim_task]
        :return: a list of tuple (0, manager stop)
        """
        list_data = []
        nb_batch = input.shape[0]

        for i in range(0,nb_batch):
            manager_stop = stops_manager_cvrp.StopsManagerCRVP.from_array(input[i,:,:])
            list_data.append((1,manager_stop))

        return list_data

    @staticmethod
    def _parse_txt_input(path):
        """
        Read the txt input, decode it
        :param path: the path of the input
        :return: a list of tuple (nb_vehi, manager stop)
        """
        list_data = []
        data_txt = open(path,'r')
        line = data_txt.readline()


        while line:
            nb_vehi = int(line.split('*')[0])
            # manager_stop = stops_manager_cvrptw.StopsManagerCVRPTW.from_txt_transfer(line)
            manager_stop = stops_manager_cvrp.StopsManagerCRVP.from_txt_transfer(line)
            list_data.append((nb_vehi,manager_stop))

            line = data_txt.readline()

        data_txt.close()

        return list_data


    @staticmethod
    def _parse_txt_input_to_array(path):
        """
        Read the txt input, decode it
        :param path: the path of the input
        :return: the array corresponding to it
        """
        nb_line = TransferDatasetUtils.file_len(path)

        data_txt = open(path,'r')
        line = data_txt.readline()
        input_path = np.zeros(shape=[nb_line,11,3])
        list_label = []
        comp = 0
        while line:
            nb_vehi = int(line.split('*')[0])
            list_label.append(nb_vehi - 1)

            # create the depot
            depot_section = line.split('*')[1]
            words = depot_section.split('-')
            input_path[comp,0,:] = [words[0],words[1],0.0]

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
                        if 'e' in w:
                            # find the indice i
                            for t,nu in enumerate(st_w):
                                if 'e' in nu:
                                    break
                            else:
                                assert False
                            st_w[t] = 0
                            st_w.pop(t+1)
                        input_path[comp,i+1,:] = np.array([st_w[0],st_w[1],st_w[2]])

            line = data_txt.readline()
            comp +=1

        data_txt.close()

        return list_label,input_path

    @staticmethod
    def file_len(fname):
        i = 0
        with open(fname) as f:
            for i, l in enumerate(f):
                pass

        return i + 1

    def _find_type(self,time):
        """
        Given a certain time, find the category to which it belongs.
        :param time: the time needed to travel
        :return: a type, or -1 if none
        """
        for i,test in enumerate(self._scale):
            if time <= test:
                return i

        return -1


    def parse_tuple(self,t):
        """
        Parse one tupel (nb_vehi, manager_stop). Update corresponding self attribute
        :param t: the corresponding tuple.
        """
        if self._true_dist:
            self._parse_tuple_dist(t)
        else:
            self._parse_tuple_edge_type(t)


    def _parse_tuple_dist(self,t):
        """
        Parse one tuple. Update corresponding self attribute
        :param t: the corresponding tuple.
        """
        edge_type = 0

        conversion_matrix = {}
        # Add the label
        true_label = min(self.MAX_VEHI, t[0]) - 1
        # one_hot_encoded = [0.0 for i in range(0, self.MAX_VEHI)]
        # one_hot_encoded[true_label] = 1.0
        self.list_labels.append(true_label)   # we have to put them between 0 and max -1

        stop_manager = t[1]

        # Create features and dist matrix, including the depot at position 0
        features = []
        matrix_type_num = np.zeros(shape=(1,len(stop_manager) +1))
        dist_matrix = [[]]      # corresponds to only one type, keeping the notion of type mostly to fit on the rest

        # depot
        features.append(stop_manager.depot.features)
        for stop in stop_manager.values():
            conversion_matrix[stop.guid] = len(conversion_matrix) +1
            test_dist = stop.get_distance_to_another_stop(stop_manager.depot)
            # if test_dist <=0.1:
            #     test_dist = max_weight
            # else:
            #     test_dist = 100/test_dist

            dist_matrix[edge_type].append((0, conversion_matrix[stop.guid],test_dist))
            matrix_type_num[edge_type, conversion_matrix[stop.guid]] +=1

        for stopId in stop_manager:
            stop = stop_manager[stopId]
            features.append(stop.features)
            dist_stop = stop.get_distance_to_another_stop(stop_manager.depot)
            # if dist_stop <= 0.1:
            #     dist_stop = max_weight
            # else:
            #     dist_stop = 100/dist_stop

            dist_matrix[edge_type].append((conversion_matrix[stopId],0,dist_stop))
            matrix_type_num[edge_type, 0] +=1

            for stopId_2 in stop_manager:
                if stopId != stopId_2:
                    dist_stop = stop.get_distance_to_another_stop(stop_manager[stopId_2])
                    # if dist_stop <= 0.1:
                    #     dist_stop = max_weight
                    # else:
                    #     dist_stop = 100/dist_stop

                    dist_matrix[edge_type].append((conversion_matrix[stopId],conversion_matrix[stopId_2],dist_stop))
                    matrix_type_num[edge_type, conversion_matrix[stopId_2]] +=1

        type_to_adj_list = [np.array(sorted(adj_list)) if len(adj_list) > 0 else np.zeros(shape=(0,3), dtype=np.float32)
                        for adj_list in dist_matrix]

        assert np.array(type_to_adj_list).shape[0] == 1, print(np.array(type_to_adj_list).shape[0])

        self.list_features.append(features)
        self.list_dist_matrix.append(type_to_adj_list)
        self.list_type_num.append(matrix_type_num)



    def _parse_tuple_edge_type(self,t):
        """
        Parse one tuple. Update corresponding self attribute
        :param t: the corresponding tuple.
        """
        conversion_matrix = {}
        # Add the label
        true_label = min(self.MAX_VEHI, t[0]) - 1
        # one_hot_encoded = [0.0 for i in range(0, self.MAX_VEHI)]
        # one_hot_encoded[true_label] = 1.0
        self.list_labels.append(true_label)   # we have to put them between 0 and number vehi -1

        stop_manager = t[1]

        # Create features and dist matrix, including the depot at position 0
        features = []
        matrix_type_num = np.zeros(shape=(self._number_type,len(stop_manager) +1))
        dist_matrix = [[] for _ in range(0,self._number_type)]

        # depot
        features.append(list(stop_manager.depot.features))
        for stop in stop_manager.values():
            conversion_matrix[stop.guid] = len(conversion_matrix) +1
            test_dist = stop.get_distance_to_another_stop(stop_manager.depot)
            edge_type = self._find_type(test_dist)
            if edge_type != -1:
                dist_matrix[edge_type].append((0, conversion_matrix[stop.guid],test_dist))
                matrix_type_num[edge_type, conversion_matrix[stop.guid]] +=1

        for stopId in stop_manager:
            stop = stop_manager[stopId]
            features.append(list(stop.features))
            dist_stop = stop.get_distance_to_another_stop(stop_manager.depot)
            edge_type = self._find_type(dist_stop)
            if edge_type != -1:
                dist_matrix[edge_type].append((conversion_matrix[stopId],0,dist_stop))
                matrix_type_num[edge_type, 0] +=1

            for stopId_2 in stop_manager:
                if stopId != stopId_2:
                    dist_stop = stop.get_distance_to_another_stop(stop_manager[stopId_2])
                    edge_type = self._find_type(dist_stop)
                    if edge_type != -1:
                        dist_matrix[edge_type].append((conversion_matrix[stopId],conversion_matrix[stopId_2],dist_stop))
                        matrix_type_num[edge_type, conversion_matrix[stopId_2]] +=1

        type_to_adj_list = [np.array(sorted(adj_list), dtype=np.float32) if len(adj_list) > 0 else np.zeros(shape=(0,3), dtype=np.float32)
                        for adj_list in dist_matrix]

        assert np.array(type_to_adj_list).shape[0] == self._number_type, print(np.array(type_to_adj_list).shape[0])
        self.list_features.append(features)
        self.list_dist_matrix.append(type_to_adj_list)
        self.list_type_num.append(matrix_type_num)


    def normalize_inverse_distance(self):
        """
        Normalize all distance by edge type
        :return:
        """
        nb_graph = len(self.list_dist_matrix)

        list_final_normalize = []
        for i in range(0,self._number_type):
            all_distances = [adj_list[i][:,2] for adj_list in self.list_dist_matrix]
            all_distances = [1/max(0.1,dist) for l in all_distances for dist in l]
            all_graph_number = [len(adj_list[i][:,2]) for adj_list in self.list_dist_matrix]
            all_distances = np.array(all_distances)

            # we reshape to norm
            list_to_norm = all_distances.reshape(-1,1)
            if len(list_to_norm) > 0:
                list_to_norm = normalize(list_to_norm,axis=0)
                max_norm = max(list_to_norm)
                list_to_norm = np.array([elt/max_norm for elt in list_to_norm])
            list_to_norm = list_to_norm.reshape(1,-1)[0]

            new_distance = []
            offset = 0
            for k in range(0,nb_graph):
                new_list = []
                for j in range(0,all_graph_number[k]):
                    new_list.append(list_to_norm[offset])
                    offset +=1

                new_distance.append(new_list)
            list_final_normalize.append(new_distance)

        # override
        for k in range(0,nb_graph):
            for i in range(0,self._number_type):
                self.list_dist_matrix[k][i][:,2] = list_final_normalize[i][k]


    def normalize_features(self):
        """
        Normalize across all instances the matrix of pixel
        :return: a list of normalized pixel matrix
        """
        total_nb_graph = len(self.list_features)
        list_node_graph = [len(graph) for graph in self.list_features]

        list_normalized_features = []

        for f in range(0,self.number_features):
            initial = []
            for k in range(0,total_nb_graph):
                initial.extend([features[f] for features in self.list_features[k]])

            initial = np.array(initial)
            # we reshape
            list_to_norm = initial.reshape(-1,1)
            list_to_norm = normalize(list_to_norm,axis=0)
            list_to_norm = list_to_norm.reshape(1,-1)[0]
            list_normalized_features.append(list_to_norm)


        # Recreate the list of normalized features
        final_list = []
        offset = 0
        for k in range(0,total_nb_graph):
            final_features = []
            for s in range(0,list_node_graph[k]):
                features_stop = []
                for f in range(0,self.number_features):
                    features_stop.append(list_normalized_features[f][offset])
                offset +=1
                final_features.append(features_stop)

            final_list.append(final_features)

        assert len(final_list) == total_nb_graph

        self.list_features = final_list


    def split_data(self):
        """
        Split the data into three parts, train, valid, and test
        :return: a dict[train] = list dist matrix, a dict[train] = list_feature, a dict[train] = list label
        """
        total_data = len(self.list_labels)
        first_bound = int(0.7 * total_data)
        second_bound = int(0.85 * total_data)

        dict_matrix = {DataFold.TRAIN : self.list_dist_matrix[0:first_bound],
                       DataFold.VALIDATION : self.list_dist_matrix[first_bound:second_bound],
                       DataFold.TEST : self.list_dist_matrix[second_bound:]}

        dict_type_enum = {DataFold.TRAIN : self.list_type_num[0:first_bound],
                       DataFold.VALIDATION : self.list_type_num[first_bound:second_bound],
                       DataFold.TEST : self.list_type_num[second_bound:]}


        dict_features = {DataFold.TRAIN : self.list_features[0:first_bound],
                       DataFold.VALIDATION : self.list_features[first_bound:second_bound],
                       DataFold.TEST : self.list_features[second_bound:]}

        dict_labels = {DataFold.TRAIN : self.list_labels[0:first_bound],
                       DataFold.VALIDATION : self.list_labels[first_bound:second_bound],
                       DataFold.TEST : self.list_labels[second_bound:]}

        return dict_matrix, dict_type_enum, dict_features, dict_labels



