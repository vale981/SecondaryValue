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

        self._parsed_lambda = sympy.lambdify(self._parsed.free_symbols,
                                             self._expr)

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
            kwargs[name] = tmp
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

    def _calculate_gauss_propagation(self, values, derivs, error):
        """Calculates a single gaussian error propagation.

        :returns: error
        :rtype: dtype
        """

        term = np.array([(derivs[var](**values) * err) \
                          for var, err in error.items() \
                          if err > 0], dtype=self._dtype)

        term = np.sqrt(term.dot(term))

        return self._dtype(term)

    def _process_args(self, *args, **kwargs):
        """Process the deconstruct given to `__call__`.

        :returns: values (input values), errors (their errors),
                  dep_values (the values and errors of the dependencies)

        :rtype: Tuple
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


        max_uncertainties = max([len(val) for _, val in kwargs.items() \
                                 if isinstance(val, Iterable)] or [0])

        # filter out the error values
        errors = [{var: val[i] for var, val in kwargs.items() \
                   if isinstance(val, Iterable) and len(val) > i} \
                  for i in range(1, max_uncertainties)]


        values = {var: (val[0] if isinstance(val, Iterable) else val) \
                  for var, val in kwargs.items()}

        return values, errors, dep_values

    def _calculate_central_value(self, scalar_values, vector_values):
        """Calculate the central value from the scalar and/or
        vectorized input values.

        :param dict scalar_values: the scalar input variables
        :param dict vector_values: the vectorized input variables

        :returns: the central value or an array of central values
        :rtype: dtype / np.array[dtype]
        """
        central_value = 0
        value_length = max([len(elem) \
                            for elem in vector_values.values()] or [0])

        if vector_values:
            central_value = np.empty(value_length)
            for i in range(0, value_length):
                current_values = join_row(scalar_values, vector_values, i)

                central_value[i] = self._parsed_lambda(**current_values)
        else:
            central_value = self._dtype(self._parsed_lambda(**scalar_values))

        return central_value

    def _calculate_errors(self, errors, vector_values, scalar_values):
        """Calculate the errors for the secondary value.

        :param list[dict] errors: A list of dictionaries containing the errors.
        :param dict vector_values: the central values
        :param dict scalar_values: the scalar central values
        :returns: error or list or errors
        """

        derivs = self._get_derivatives(*list(errors[0].keys()))
        terms = []

        # iterate error series (horizontal)
        for error in errors:
            scalar_errors, vector_errors = filter_out_vecotrized(error)
            length = max([len(elem) \
                          for elem in (list(vector_values.values())
                                      + list(vector_errors.values()))] or [0])

            # if there are only scalar values and errors
            if length == 0:
                terms.append(self._calculate_gauss_propagation(scalar_values,
                                         derivs, error))

            # calculate error for every (value, error) pair. Errors
            # are padded.
            else:
                tmp = np.empty(length, dtype=self._dtype)
                for i in range(0, length):
                    current_values = join_row(scalar_values, vector_values, i)
                    current_errors = join_row(scalar_errors, vector_errors, i)

                    tmp[i] = \
                        self._calculate_gauss_propagation(current_values,
                                                          derivs, current_errors)
                terms.append(tmp)

        return terms

    def __call__(self, *args, **kwargs):
        """Calculates a value from the expression by substituting
        variables by the values of the given keyword arguments.  If an
        argument is specified as a tuplpe of (value, error) the
        gausssian error propagation will be computed.

        The values and errors can be iterable, but must compatible shapes.

        :returns: value or [value, error] or [value, error], dependencies

        :rtype: numpy data type or np array of [value, errors, ...] or
                a tuple the beforementioned as first element and a
                dictionary with the calculated dependencies as a second value
        """

        # process the keyword arguments
        values, errors, dep_values = self._process_args(*args, **kwargs)

        # calulate the central value
        scalar_values, vector_values = filter_out_vecotrized(values)
        central_value = self._calculate_central_value(scalar_values,
                                                      vector_values)

        if not errors:
            return central_value

        # calculate errors
        result = self._calculate_errors(errors, vector_values, scalar_values)

        # create the result tuple
        result.insert(0, central_value)
        result = tuple(result)

        if dep_values:
            return result, dep_values

        return result


    @lru_cache(maxsize=32)
    def _get_derivatives(self, *args):
        """Calculates the derivatives of the expression for a given
        set of variables specified by args.
        """

        for var in args:
            if var not in self._derivatives:
                self._derivatives[var] = \
                    sympy.lambdify(self._parsed.free_symbols
                                   , diff(self._parsed, var))

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


def filter_out_vecotrized(dictionary):
    scalar = dict()
    vector = dict()

    for key, value in dictionary.items():
        if isinstance(value, Iterable):
            vector[key] = value
        else:
            scalar[key] = value

    return scalar, vector

def join_row(scalar, vector, index):
     return {**scalar, **{key: val[index] \
                          for key, val in vector.items()}}
