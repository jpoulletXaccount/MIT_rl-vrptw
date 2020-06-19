
import numpy as np

class Stop(object):
    """
    Baseline class of stops
    """

    def __init__(self,guid,x,y):
        self.guid = guid
        self.x = float(x)
        self.y = float(y)

    @property
    def coordinates(self):
        return self.x,self.y

    @property
    def no_id(self):
        return int(self.guid.split("_")[1])

    @property
    def xy(self):
        return self.x,self.y


    def get_distance_to_another_stop(self,stop):
        """
        Compute the distance with an other stop
        :param stop: the other stop
        :return: the euclidiean distance
        """
        other_x,other_y = stop.xy
        a= np.array(self.xy)
        b = np.array([other_x,other_y])
        return np.linalg.norm(a - b)
