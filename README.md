# Secondary Value
[![Coverage Status](https://coveralls.io/repos/github/vale981/SecondaryValue/badge.svg)](https://coveralls.io/github/vale981/SecondaryValue)
[![Build Status](https://travis-ci.org/vale981/SecondaryValue.svg?branch=master)](https://travis-ci.org/vale981/SecondaryValue)

This is a utility to simplify the calculation of values and their
uncertaintues from symbolic formulas by using `sympy` and `numpy`.

## Installation
Just a quick `pip install SecondaryValue` will do.

## Examples
(A Documentation will follow soon. For now: look at the docstrings!)

### Basic Usage
```python
from SecondaryValue import SecondaryValue

# create a secondary value
# the argument can be either a string or a sympy expression
x = SecondaryValue("a*b/sqrt(c)")

# Calculate a result value by substi3tuting the keyword arguments
# where a keyword agument may consist of (value, error_1, error_2, ...)
# and (...) stands for any iterable.
result = x(a=(1, 20), b=(2,  30), c=2)

# The calculation returns a numpy array with the length of the longest
# keyword argument above: [value, error_1, error_2]
# For each error_n the uncertainties error_n of the keyword args above are
# used if present. This may be useful to calculate statistical and systemic
# errors in one go.
print(result)
# >> (1.41421356, 35.35533906)

# As a goodie, you can print out the gaussian error distribution in
# symbolic form. (Works best in Jupyter Notebooks)
x.pretty_gauss_propagation('a', 'b', 'c')
```

### Default Values
To reduce boilerplate one can set default substitutions for symbols (with errors).
This especially useful for constants.

```python
from SecondaryValue import SecondaryValue

# create a secondary value with default arguments
x = SecondaryValue("a + b", defaults=dict(b=1/2))

# this works because `b` is substituted from the defaults
result = x(b=1/2)
print(result)
# >> 1.0

# As a goodie, you can print out the gaussian error distribution in
# symbolic form. (Works best in Jupyter Notebooks)
x.pretty_gauss_propagation('a', 'b', 'c')
```

### Vectorized Input
`SecondaryValue` supports vectorized input. As a rule-of-thump: Put
the iterable (list, np.array) where you would put scalars.

You can mix scalars and vectors as long as all errors and values are
either scalar or have the same length.

```python
from SecondaryValue import SecondaryValue
x = SecondaryValue('a**2+b')

x(a=[[1,2,3]], b=1)
# >> array([ 2.,  5., 10.])

x(a=([1,2,3], 1), b=1)
# >> (array([ 2.,  5., 10.]), array([2., 4., 6.]))

x(a=([1,2,3], [1,2,3]), b=1)
# >> (array([ 2.,  5., 10.]), array([ 2.,  8., 18.]))

x(a=([1,2,3], [1,2,3]), b=([1,2,3], 1))
# >> (array([ 2.,  6., 12.]), array([ 2.23606798,  8.06225775, 18.02775638]))

# THAT DOES NOT WORK:
x(a=([1,2,3], [1,2,3]), b=([1,2], 1))
```

If all the returned arrays in the tuple have the same shape, you can
easily convert that tuple to a numpy array:
`np.array(x(a=([1,2,3], [1,2,3]), b=([1,2,3], 1)))`

### Dependencies
To make the calculation of complex values easier, one can define
dependencies for a `SecondaryValue`:

```python
from SecondaryValue import SecondaryValue

dep = SecondaryValue('u')
x = SecondaryValue("a + b", dependencies=dict(a=dep))

# x will now accept u as an additional kwarg and calculate d==dep on the fly
# and return a dictionary containing it as a second return value if you
# specify `retdeps=True`.
print(x(b=1, u=(1, 2), retdeps=True))
# >> ((2.0, 2.0), {'a': ((1.0, 2.0), {})})

# To make the output predictable, the dependencies aren't returned by deafult.
print(x(b=1, u=(1, 2)))
# >> (2.0, 2.0)

# you can overwrite the dependency calculation
print(x(b=1/2, a=1/2))
# >> 1.0
```

If there are no dependency values, an empty dict will be returned when
`retdeps=True` is specified.
