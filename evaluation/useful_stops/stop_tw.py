
from evaluation.useful_stops import stop

class StopTW(stop.Stop):
    """
    Instance of a stop with time window
    """
    def __init__(self,guid,x,y,demand, begin_tw,end_tw):
        super(StopTW,self).__init__(guid,x,y,demand)
        self.begin_tw = float(begin_tw)
        self.end_tw = float(end_tw)


        assert self.begin_tw < self.end_tw, print(begin_tw, end_tw)

    @property
    def tw(self):
        return self.begin_tw,self.end_tw


    def to_print(self):
        txt = "stop_tw, " + str(self.x) + " " + str(self.y) + " "
        txt += str(self.begin_tw) + "  " + str(self.end_tw) + " " + str(self.demand)
        return txt

