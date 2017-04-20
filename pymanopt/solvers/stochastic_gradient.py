from __future__ import print_function, division

import time
from copy import deepcopy

import numpy as np

from pymanopt.solvers.linesearch import LineSearchBackTracking
from pymanopt.solvers.solver import Solver

from pymanopt.tools.autodiff._utils import flatten, unflatten

class SGD(Solver):
    """
    Steepest descent (gradient descent) algorithm based on
    steepestdescent.m from the manopt MATLAB package.
    """

    def __init__(self, linesearch=LineSearchBackTracking(), *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

        if linesearch is None:
            self._linesearch = StochasticLineSearchBackTracking()
        else:
            self._linesearch = linesearch
        self.linesearch = None

    # Function to solve optimisation problem using steepest descent.
    def solve(self, problem, e_problem, dataset, batch_size, learning_rate_starter,
                 learning_rate_decay_steps,learning_rate_decay_rate, w=None, reuselinesearch=False):
        """
        Perform optimization using gradient descent with linesearch.
        This method first computes the gradient (derivative) of obj
        w.r.t. arg, and then optimizes by moving in the direction of
        steepest descent (which is the opposite direction to the gradient).
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .manifold attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
            - x=None
                Optional parameter. Starting point on the manifold. If none
                then a starting point will be randomly generated.
            - reuselinesearch=False
                Whether to reuse the previous linesearch object. Allows to
                use information from a previous solve run.
            - dataset
                should have a method:  batch_xs, batch_ys = data.next_batch(n)
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
        """
        man = problem.manifold
        e_man = e_problem.manifold
        verbosity = problem.verbosity
        objective = problem.cost
        e_objective = e_problem.cost
        gradient = problem.grad
        e_gradient = e_problem.grad
        accuracy_and_summary = problem.accuracy_and_summary

        learning_rate = learning_rate_starter

        if not reuselinesearch or self.linesearch is None:
            self.linesearch = deepcopy(self._linesearch)
        linesearch = self.linesearch

        # If no starting point is specified, generate one at random.
        if w is None:
            w = man.rand()


        # Initialize iteration counter and timer
        iter = 0
        time0 = time.time()

        if verbosity >= 1:
            print(" iter\t\t   cost\t        grad.norm\t    cost test\t     accuracy test\t     euc diff")

        self._start_optlog(extraiterfields=['gradnorm'],
                           solverparams={'linesearcher': linesearch})

        while True:
            iter = iter + 1
            if len(w[0])==3:
                e_w = [w[0][0].dot(np.diag(w[0][1])).dot(w[0][2]),w[1]]
            else:
                e_w = [w[0][0].dot(w[0][1]), w[1]]

            if iter % learning_rate_decay_steps == 0:
                learning_rate *= learning_rate_decay_rate

            if iter % 10 == 0:
                euc_diff = max(np.abs(e_new_w[0] - e_w[0]).max(),
                               np.abs(e_new_w[1] - e_w[1]).max())
                data_test =  [dataset.test.images, dataset.test.labels]
                cost_test = objective(w + data_test)
                accu_test, summary = accuracy_and_summary(w + data_test)
                if verbosity >= 1:
                    print("%5d\t%+.16e\t%.8e\t%+.16e\t%.2f\t%+.16e" % (iter,  cost, gradnorm, cost_test, accu_test,euc_diff))
                problem.write_summary(summary,iter)

            else:
                # Calculate new cost, grad and gradnorm
                batch_xs, batch_ys = dataset.train.next_batch(batch_size)
                data = [batch_xs, batch_ys]

                cost = objective(w+data)
                e_cost = e_objective(e_w + data)
                grad, egrad = gradient(w+data)
                e_grad, e_egrad = e_gradient(e_w+data)

                #amb = man._manifolds[0].tangent2ambient(w[0], grad[0])
                #tangent_grad = amb[0].dot(amb[1]).dot(amb[2].T)

                gradnorm = man.norm(w, grad)

                if verbosity >= 2:
                    print("%5d\t%+.16e\t%.8e" % (iter, cost, gradnorm))

                if self._logverbosity >= 2:
                    self._append_optlog(iter, w, cost, gradnorm=gradnorm)

                # for debug calulate euclidian update
                new_ew = [x-learning_rate*d for x,d in zip(flatten(w),flatten(egrad))]

                # Descent direction is minus the gradient
                desc_dir = -grad
                e_desc_dir = -e_grad

                # update w
                new_w = man.retr(w, learning_rate  * desc_dir)
                e_new_w = e_man.retr(e_w, learning_rate  * e_desc_dir)

                dbg = [np.abs(a-b).mean() for a,b in zip(flatten(new_w),new_ew)]

                w = new_w
                stepsize = man.norm(w, desc_dir)


                # Perform line-search
                #stepsize, w = linesearch.search(objective, man, w, data, desc_dir,
                #                                cost, -gradnorm**2)

                stop_reason = self._check_stopping_criterion(
                    time0, stepsize=stepsize, gradnorm=gradnorm, iter=iter)


                if stop_reason:
                    if verbosity >= 1:
                        print(stop_reason)
                        print('')
                    break


        if self._logverbosity <= 0:
            return w
        else:
            self._stop_optlog(w, objective(w+data), stop_reason, time0,
                              stepsize=stepsize, gradnorm=gradnorm,
                              iter=iter)
            return w, self._optlog
