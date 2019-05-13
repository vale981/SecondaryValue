# Secondary Value

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

# Calculate a result value by substituting the keyword arguments
# where a keyword agument may consist of (value, error_1, error_2, ...)
# and (...) stands for any iterable.
result = x(a=(1, 20), b=(2,  30), c=2)

# The calculation returns a numpy array with the length of the longest
# keyword argument above: [value, error_1, error_2]
# For each error_n the uncertainties error_n of the keyword args above are
# used if present. This may be useful to calculate statistical and systemic
# errors in one go.
print(result)
# >> array([ 1.41421356, 35.35533906])

# As a goodie, you can print out the gaussian error distribution in
# symbolic form. (Works best in Jupyter Notebooks)
x.pretty_gauss_propagation('a', 'b', 'c')
```
### Dependencies
To make the calculation of complex values easier, one can define
dependencies for a `SecondaryValue`:

```python
from SecondaryValue import SecondaryValue

dep = SecondaryValue('u')
x = SecondaryValue("a + b", dependencies={'a': dep})

# x will now accept u as an additional kwarg and calculate d==dep on the fly
# and return a dictionary containing it as a second return value.
print(x(b=1, u=(1, 2)))
# >> (array([2., 2.]), {'a': array([1., 2.])})

# you can overwrite the dependency calculation
print(x(b=1/2, a=1/2))
# >> 1.0
```
