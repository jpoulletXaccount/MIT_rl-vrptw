from shared.graph_embedding.useful_stops import stops_cvrp


class Stop_cvrptw(stops_cvrp.Stop_cvrp):
    """
    Class of stop corresponding to the capacitated vehicle routing problem with time window
    """
    def __init__(self,guid,x,y,demand,beginTW,endTW,service_time):
        super(Stop_cvrptw,self).__init__(guid,x,y,demand)
        self.beginTW = float(beginTW)
        self.endTW = float(endTW)
        self.service_time = int(service_time)
        self.x = float(x)
        self.y = float(y)


    @property
    def TW(self):
        return self.beginTW,self.endTW

    @property
    def features(self):
        return self.x,self.y, self.demand, self.beginTW, self.endTW, self.service_time
        # return self.demand, self.beginTW, self.endTW, self.service_time


