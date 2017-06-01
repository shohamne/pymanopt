from __future__ import print_function, division

import time
from copy import deepcopy

import numpy as np
import pandas as pd

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
    def solve(self, problem, dataset, batch_size, learning_rate_starter,
                 learning_rate_decay_steps,learning_rate_decay_rate,epoch=10,w=None, reuselinesearch=False):
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

        verbosity = problem.verbosity
        objective = problem.cost
        argument = problem.argument
        gradient = problem.grad
        train_scalars_and_summary = problem.train_scalars_and_summary
        test_scalars_and_summary = problem.test_scalars_and_summary

        learning_rate = learning_rate_starter

        if not reuselinesearch or self.linesearch is None:
            self.linesearch = deepcopy(self._linesearch)
        linesearch = self.linesearch

        # If no starting point is specified, generate one at random.

        w = man.rand()
        w = unflatten(argument(), w)
        #if w is None:
        #    w = w_


        # Initialize iteration counter and timer
        iter = 0
        time0 = time.time()

        self._start_optlog(extraiterfields=['gradnorm'],
                           solverparams={'linesearcher': linesearch})

        print_title = True
        times = None
        iir_factor = 0.9
        cost_avg = 0
        accu_avg = 0
        cost_dropout_avg = 0
        accu_dropout_avg = 0
        while True:
            times_line = pd.Series()
            times_line['0-read'] = time.time()-time0
            if iter % learning_rate_decay_steps == 0:
                learning_rate *= learning_rate_decay_rate

            # Calculate new cost, grad and gradnorm
            batch_xs, batch_ys = dataset.train.next_batch(batch_size)
            data = [batch_xs, batch_ys]


            times_line['1-summ'] = time.time()-time0
            cost, accu, cost_dropout, accu_dropout, summary = train_scalars_and_summary(w+data)

            cost_avg = cost_avg*iir_factor + cost*(1-iir_factor)
            accu_avg = accu_avg*iir_factor + accu*(1-iir_factor)
            cost_dropout_avg = cost_dropout_avg*iir_factor + cost_dropout*(1-iir_factor)
            accu_dropout_avg = accu_dropout_avg * iir_factor + accu_dropout * (1 - iir_factor)

            times_line['2-summ'] = time.time()-time0
            problem.write_summary(summary, iter)


            times_line['3-grad'] = time.time()-time0
            grad, egrad = gradient(w,data)


            times_line['4-norm'] = time.time()-time0
            gradnorm = man.norm(w, grad)


            times_line['5-eval'] = time.time()-time0
            if iter % epoch == 0 and iter > 0:
                data_test =  [dataset.test.images, dataset.test.labels]
                cost_test, accu_test, cost_dropout_test, accu_dropout_test, summary_test = \
                    train_scalars_and_summary(w + data_test)

                problem.write_summary(summary_test,iter)

                times_diff = {}
                if times is not None:
                    times_diff = times.copy()[:-1]
                    times_diff.values[:] -= times.values[1:]
                    times_diff = -times_diff
                    times_diff = times_diff/times_diff.sum()
                    #times_diff = pd.DataFrame(-times_diff).T
                    #times_diff.index = ['tm']

                if verbosity >= 1 and print_title:
                    print("iter\trate\tcost\taccu\tcost_avg\taccu_avg\t"
                          "cost_do\t\taccu_do\t\tcost_do_avg\taccu_do_avg\t"
                          "grad_norm\tcost_test\taccu_test\tcost_do_test\taccu_do_test",end='')
                    for i in times_diff.index:
                        print("%s" % i,end='\t')
                    print()
                    print_title = False

                if verbosity >= 1:
                    print("%5d\t%.4f\t%.4f\t%.4f\t%.4f\t\t%.4f\t\t"
                          "%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t"
                          "%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t\t%.4f\t\t" %
                          (iter, learning_rate, cost, accu, cost_avg, accu_avg,
                           cost_dropout, accu_dropout,cost_dropout_avg, accu_dropout_avg,
                           gradnorm, cost_test, accu_test, cost_dropout_test, accu_dropout_test),end='')
                    for i in times_diff.values:
                        print("%.2f" % i,end='\t')
                    print()


            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e" % (iter, cost, gradnorm))

            if self._logverbosity >= 2:
                self._append_optlog(iter, w, cost, gradnorm=gradnorm)



            times_line['6-apply'] = time.time()-time0
            # Descent direction is minus the gradient
            desc_dir = -grad


            # update w
            w = man.retr(w, learning_rate  * desc_dir)
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

            times_line['7-end'] = time.time()-time0

            if times is not None:
                times += times_line
            else:
                times = times_line

            iter += 1


        if self._logverbosity <= 0:
            return w
        else:
            self._stop_optlog(w, objective(w+data), stop_reason, time0,
                              stepsize=stepsize, gradnorm=gradnorm,
                              iter=iter)
            return w, self._optlog
