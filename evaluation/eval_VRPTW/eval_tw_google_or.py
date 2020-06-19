
from evaluation.eval_VRPTW import evaluator,google_tw_solver

class EvalTWGoogleOR(evaluator.RoutingEvaluator):
    """
    Using google or tools, eval the VRPTW porblem
    """

    def __init__(self,args,env,prt,min_veh):
        super(EvalTWGoogleOR,self).__init__(args,env,prt,min_veh)
        self.name = 'or_tools_tw'

        self._update_filename(self.name)


    def _route_creator(self, manager):
        """
        Route a manager and output the sequence in a specific file
        :param manager: the considered manager
        :return: a sequence of stop
        """
        solver_object = google_tw_solver.GoogleSolverTW(manager_stop_tw=manager,
                                                       env=self.env,min_veh=self.min_veh)
        return solver_object.solve()
