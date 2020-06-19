
import numpy as np

class Stop(object):
    """
    Instance of a stop
    """
    def __init__(self,guid,x,y,demand):
        self.guid = guid
        self.x = float(x)
        self.y = float(y)
        self.demand = int(float(demand))

        assert abs(self.demand - float(demand)) <= 0.0001, demand


    @property
    def xy(self):
        return self.x, self.y


    def get_dist_another_stop(self,stop):
        """
        :param stop: an other stop
        :return: the euclidean distance
        """
        dist = np.sqrt((self.x - stop.x)**2 + (self.y - stop.y)**2)
        return dist

