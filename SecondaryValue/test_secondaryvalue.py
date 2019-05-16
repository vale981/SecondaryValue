from SecondaryValue import SecondaryValue
import random
from pytest import approx, raises
import numpy

class TestBasicUsage():
    def setup(self):
        self.x = SecondaryValue('a*b+c')

    def test_no_uncertainties(self):
        # no uncertainties
        assert 1.2 * 7.9 + 10 == self.x(a=1.2, b=7.9, c=10)

    def test_get_symbols(self):
        # the right symbols
        assert self.x.get_symbols() == {'a': {}, 'b': {}, 'c': {}}

    def ret_deps_without_deps(self):
        # retdeps returns a dict
        assert self.x(a=1, b=1, c=1, retdeps=True)[1] == {}

    def test_uncertainties(self):
        # dirty
        for i in range(100):
            a, b, c = [(random.random(), random.random()) for _ in range(3)]
            assert (a[0]*b[0] + c[0], numpy.sqrt((a[0]*b[1])**2 + (a[1]*b[0])**2 + c[1]**2)) \
                == approx(self.x(a=a, b=b, c=c))

    def test_error_missing(self):
        with raises(RuntimeError):
            self.x(u=1)

class TestDependecies():
    def setup(self):
        self.x = SecondaryValue('b')
        self.y = SecondaryValue('a + x', dependencies=dict(x=self.x))

    def test_overwrite(self):
        result, deps = self.y(a=1, x=1, retdeps=True)
        assert deps == {}
        assert result == 2

    def test_dep_calc(self):
        result, deps = self.y(a=1, b=2, retdeps=True)

        assert result == 3
        assert deps == {'x': (2, {})}

    def test_dep_calc_err(self):
        # no retdep
        for i in range(100):
            a, b = [(random.random(), random.random()) for _ in range(2)]
            result, deps = self.y(a=a, b=b, retdeps=True)

            assert (a[0]+b[0], numpy.sqrt((b[1])**2 + (a[1])**2)) \
                == approx(result)

            assert (b[0], numpy.abs(b[1])) == approx(deps['x'][0])


class TestDefaults():
    def setup(self):
        self.y = SecondaryValue('a + b', defaults=dict(a=(1, 2)))

    def test_basics(self):
        assert (2, 2) == self.y(b=1)

class TestShapes():
    def setup(self):
        self.x = SecondaryValue('b')
        self.y = SecondaryValue('a + x', dependencies=dict(x=self.x))

    def test_array_central_value(self):
        # mixed
        assert numpy.any(numpy.array([1,2,3]) == self.y(a=([1,2,3],), b=0))

        # both
        assert numpy.any(2*numpy.array([1,2,3]) \
                         == self.y(a=([1,2,3],), b=([1,2,3],)))

#    def test_array_uncert(self):
