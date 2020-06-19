

class Depot(object):
    """
    Depot: start and end of the vehicles
    """
    def __init__(self,x,y,max_time = None):
        if abs(round(float(x)) - float(x)) <= 0.000001:
            self.x = int(x)
            self.y = int(y)
        else:
            self.x = float(x)
            self.y = float(y)
        if max_time is None:
            self._max_time = max_time
        else:
            self._max_time = float(max_time)

    @property
    def xy(self):
        return self.x,self.y

    @property
    def due_date(self):
        if self._max_time is None:
            assert False
        return self._max_time


    @property
    def features(self):
        # return self.x, self.y,0, 0,0,0, self._max_time, 0
        # return 0, 0, self._max_time, 0
        return self.x, self.y,0, 0, self._max_time, 0


