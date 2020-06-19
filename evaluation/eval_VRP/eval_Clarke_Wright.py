
from evaluation.eval_VRP import evaluator,Clarke_Wright_Saving_algo


class EvalClarkeWright(evaluator.RoutingEvaluator):
    """Using google or tools, eval the VRP problem"""

    def __init__(self,args,env,prt,min_veh):
        super(EvalClarkeWright,self).__init__(args,env,prt,min_veh)
        self.name = 'Clarke_Wright'
        self._update_filename(self.name)


    def _route_creator(self, manager):
        """
        Route a manager and output the sequence in a specific file
        :param manager: the considered manager
        :return: a sequence of stop
        """
        solver_object = Clarke_Wright_Saving_algo.ClarkeWrightSolver(manager_stop=manager,
                                                                    env=self.env)
        return solver_object.solve()




