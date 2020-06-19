
from evaluation.eval_VRP import evaluator,google_solver


class EvalGoogleOR(evaluator.RoutingEvaluator):
    """Using google or tools, eval the VRP problem"""

    def __init__(self,args,env,prt,min_veh):
        super(EvalGoogleOR,self).__init__(args,env,prt,min_veh)
        self.name = 'or_tools'
        self._update_filename(self.name)


    def _route_creator(self, manager):
        """
        Route a manager and output the sequence in a specific file
        :param manager: the considered manager
        :return: a sequence of stop
        """
        solver_object = google_solver.GoogleSolver(manager_stop=manager,
                                                   env=self.env,min_veh=self.min_veh)
        return solver_object.solve()




