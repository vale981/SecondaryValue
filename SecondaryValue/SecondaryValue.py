"""
A Module with tools for calculating secondary values in physics lab
courses.
"""

import sympy
from functools import lru_cache
from collections.abc import Iterable
from sympy import sympify, diff, symbols
from sympy.abc import _clash

import numpy as np

class SecondaryValue:
    """
    A value computed from a formular, optionally with error propagation.
    """
    def __init__(self, expr, dependencies=False, defaults=False,
                 dtype=np.float64):
        """Creates a new Secondary Value.

        :param str expr: the expression from which to calculate the
            value, may be a sympy expression or a string.
        :param dict dependencies: secondary values to calculate as
            dependencies (in order)
        :param dict defaults: Default arguments for the call.
        :param np.dtype dtype: the numpy datatype for the resulting
            value

        """

        self._expr = expr
        self._parsed = sympify(self._expr, _clash) if isinstance(self._expr, str) \
            else self._expr

        self._symbols = {symbol.__str__() \
                         for symbol in self._parsed.free_symbols}

        self._deps = {name: dependency \
                      for name, dependency in dependencies.items() \
                      if name in self._symbols} if dependencies else {}

        self._defaults = defaults
        self._dtype = dtype

        # with per-instance cache
        self._derivatives = {}

    def __repr__(self):
        return self._expr.__repr__()

    def _calc_deps(self, **kwargs):
        """Calculate the dependencies if not present in the kwargs.

        :returns: patched kwargs containing the dependencies and a
                  filtered dict containing only the calculated dependecies

        :rtype: Tuple
        """

        calc_deps = {}
        for name, sec_val in self._deps.items():
            if name in kwargs:
                continue

            tmp = sec_val(**kwargs)
            kwargs[name] = tmp[0] if isinstance(tmp, Iterable) else tmp
            calc_deps[name] = tmp

        return kwargs, calc_deps

    def _inject_defaults(self, **kwargs):
        """Injects the default value for arguments.

        :returns: the modified argument dictionary
        :rtype: dict
        """

        if not self._defaults:
            return kwargs

        for name, value in self._defaults.items():
            if name not in kwargs:
                kwargs[name] = value

        return kwargs

    def _calculate(self, *args, **kwargs):
        """Calculates a value from the expression by substituting
        variables by the values of the given keyword arguments.  If an
        argument is specified as a tuplpe of (value, error) the
        gausssian error propagation will be computed.

        :returns: value or [value, error] or [value, error], dependencies

        :rtype: numpy data type or np array of [value, errors, ...] or
                a tuple the beforementioned as first element
        """

        max_uncertainties = max([len(val) for _, val in kwargs.items() \
                                 if isinstance(val, Iterable)] or [0])

        # filter out the error values
        errors = {var: val for var, val in kwargs.items() \
                  if isinstance(val, Iterable) and len(val) > 1}

        if not errors:
            return self._dtype(self._parsed.subs(kwargs))

        values = {var: (val[0] if isinstance(val, Iterable) else val) \
                  for var, val in kwargs.items()}

        # get them cached
        derivs = self._get_derivatives(*list(errors.keys()))

        # ugly, but works for now
        terms = [np.array([(derivs[var](values) * err[i]) \
                          for var, err in errors.items() \
                           if len(err) > i and err[i] > 0],
                          dtype=self._dtype) for i in range(1, max_uncertainties)]

        terms = np.array([np.sqrt(t.dot(t)) for t in terms], dtype=self._dtype)

        return np.insert(terms, 0, self._parsed.subs(values))

    def __call__(self, *args, **kwargs):
        """Calculates a value from the expression by substituting
        variables by the values of the given keyword arguments.  If an
        argument is specified as a tuplpe of (value, error) the
        gausssian error propagation will be computed.

        :returns: value or [value, error] or [value, error], dependencies

        :rtype: numpy data type or np array of [value, errors, ...] or
                a tuple the beforementioned as first element and a
                dictionary with the calculated dependencies as a second value
        """

        kwargs, dep_values = self._calc_deps(**kwargs)
        kwargs = self._inject_defaults(**kwargs)

        # check for missing symbols
        if not self._symbols <= set(kwargs.keys()):
            raise RuntimeError('Missing symbols: ' +
                                (self._symbols - set(kwargs.keys())).__str__())

        # filter out unneeded
        kwargs = {var: val for var, val in kwargs.items() \
                  if var in self._symbols}

        terms = self._calculate(*args, **kwargs)

        if dep_values:
            return terms, dep_values

        return terms


    @lru_cache(maxsize=32)
    def _get_derivatives(self, *args):
        """Calculates the derivatives of the expression for a given
        set of variables specified by args.
        """

        for var in args:
            if var not in self._derivatives:
                self._derivatives[var] = \
                    sympy.lambdify(args, diff(self._parsed, var))

        return {var: self._derivatives[var] for var in args}

    def pretty_gauss_propagation(self, *variables):
        """Returns a sympy expression for the gaussian error
        propagation.

        :param variables: a list of variables (strings)

        :returns: sympy expression
        """

        derivs = self._get_derivatives(*variables)
        terms = [(sympy.simplify(derivs[var]) * sympy.Dummy('Delta_' + var))**2 \
                         for var in variables]
        return sympy.sqrt(sum(terms))

    def get_symbols(self):
        """
        :returns: The symbols that can be substituted (recursively).
        :rtype: dict
        """

        return {symbol: self._deps[symbol].get_symbols() \
                if symbol in self._deps else {} \
                for symbol in self._symbols}
