"""
Module containing pymanopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""
from __future__ import print_function

from pymanopt.tools.autodiff import (AutogradBackend, TheanoBackend,
                                     TensorflowBackend)


class Problem(object):
    """
    Problem class for setting up a problem to feed to one of the
    pymanopt solvers.

    Attributes:
        - manifold
            Manifold to optimize over.
        - cost
            A callable which takes an element of manifold and returns a
            real number, or a symbolic Theano or TensorFlow expression.
            In case of a symbolic expression, the gradient (and if
            necessary the Hessian) are computed automatically if they are
            not explicitly given. We recommend you take this approach
            rather than calculating gradients and Hessians by hand.
        - grad
            grad(x) is the gradient of cost at x. This must take an
            element X of manifold and return an element of the tangent space
            to manifold at X. This is usually computed automatically and
            doesn't need to be set by the user.
        - hess
            hess(x, a) is the directional derivative of grad at x, in
            direction a. It should return an element of the tangent
            space to manifold at x.
        - egrad
            The 'Euclidean gradient', egrad(x) should return the grad of
            cost in the usual sense, i.e. egrad(x) need not lie in the
            tangent space.
        - ehess
            The 'Euclidean Hessian', ehess(x, a) should return the
            directional derivative of egrad at x in direction a. This
            need not lie in the tangent space.
        - arg
            A symbolic (tensor) variable with respect to which you would like
            to optimize. Its type (together with the type of the cost argument)
            defines the autodiff backend used.
        - verbosity (2)
            Level of information printed by the solver while it operates, 0
            is silent, 2 is most information.
    """
    def __init__(self, manifold, cost, accuracy=None,
                 cost_dropout=None, accuracy_dropout=None,
                 train_summary=None, test_summary=None,
                 egrad=None, ehess=None, grad=None,
                 hess=None, arg=None, data=[], precon=None, verbosity=2, logdir=None):
        self.manifold = manifold
        # We keep a reference to the original cost function in case we want to
        # call the `prepare` method twice (for instance, after switching from
        # a first- to second-order method).
        self._cost = None
        self._original_cost = cost
        self._train_scalars_and_summary = None
        self._original_train_scalars_and_summary = \
            [cost, accuracy, cost_dropout, accuracy_dropout, train_summary]
        self._test_scalars_and_summary = None
        self._original_test_scalars_and_summary = \
            [cost, accuracy, cost_dropout, accuracy_dropout, test_summary]
        self._argument = None
        self._egrad = egrad
        self._ehess = ehess
        self._grad = grad
        self._hess = hess
        self._arg = arg if isinstance(arg,list) else [arg]
        self._data = data
        self._backend = None
        self._logdir = logdir

        if precon is None:
            def precon(x, d):
                return d
        self.precon = precon

        self.verbosity = verbosity

        self._backends = filter(lambda b: b.is_available(),
                                [TheanoBackend(), AutogradBackend(),
                                 TensorflowBackend()])
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            for backend in self._backends:
                if backend.is_compatible(self._original_cost, self._arg):
                    if hasattr(backend, 'setup_log_writer'):
                        backend.setup_log_writer(self._logdir)
                    self._backend = backend
                    break
            else:
                backend_names = [str(backend) for backend in self._backends]
                print(backend_names)
                raise ValueError(
                    "Cannot determine autodiff backend from cost function of "
                    "type `{:s}`. Available backends are: {:s}".format(
                        self._original_cost.__class__.__name__,
                        ", ".join(backend_names)))

        return self._backend

    @property
    def cost(self):
        if self._cost is None:
            if self.verbosity >= 1:
                print("Compiling cost function...")
            self._cost = self.backend.compile_function(self._original_cost,
                                                       self._arg + self._data)
        elif self._cost is None and callable(self._original_cost):
            self._cost = self._original_cost

        return self._cost

    @property
    def train_scalars_and_summary(self):
        if self._train_scalars_and_summary is None:
            if self.verbosity >= 1:
                print("Compiling train_scalars_and_summary function...")
            self._train_scalars_and_summary = self.backend.compile_function(self._original_train_scalars_and_summary,
                                                                            self._arg + self._data)
        elif self._train_scalars_and_summary is None and callable(self._original_train_scalars_and_summary):
            self._train_scalars_and_summary = self._original_train_scalars_and_summary

        return self._train_scalars_and_summary

    @property
    def test_scalars_and_summary(self):
        if self._test_scalars_and_summary is None:
            if self.verbosity >= 1:
                print("Compiling test_scalars_and_summary function...")
            self._test_scalars_and_summary = self.backend.compile_function(self._original_test_scalars_and_summary,
                                                                           self._arg + self._data)
        elif self._test_scalars_and_summary is None and callable(self._original_test_scalars_and_summary):
            self._test_scalars_and_summary = self._original_test_scalars_and_summary

        return self._test_scalars_and_summary

    @property
    def argument(self):
        if self._argument is None:
            argument = self.backend.get_argument(self._arg)
            self._argument = argument
        return self._argument

    @property
    def egrad(self):
        if self._egrad is None:
            if self.verbosity >= 1:
                print("Computing gradient of cost function...")
            egrad = self.backend.compute_gradient(self._original_cost,
                                                  self._arg, self._data)
            self._egrad = egrad
        return self._egrad

    @property
    def grad(self):
        if self._grad is None:
            # Explicit access forces computation/compilation if necessary.
            egrad = self.egrad

            def grad(x,data=[]):
                eg = egrad(x,data)
                return self.manifold.egrad2rgrad(x, eg) , eg
            self._grad = grad
        return self._grad

    @property
    def ehess(self):
        if self._ehess is None:
            if self.verbosity >= 1:
                print("Computing Hessian of cost function...")
            ehess = self.backend.compute_hessian(self._original_cost,
                                                 self._arg)
            self._ehess = ehess
        return self._ehess

    @property
    def hess(self):
        if self._hess is None:
            # Explicit access forces computation if necessary.
            ehess = self.ehess

            def hess(x, a):
                return self.manifold.ehess2rhess(
                    x, self.egrad(x), ehess(x, a), a)
            self._hess = hess
        return self._hess

    def write_summary(self,summary,iter):
        self.backend.write_summary(summary,iter)
