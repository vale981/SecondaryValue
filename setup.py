from setuptools import setup

setup(name='SecondaryValue',
      version='0.0.3',
      description='A helper to calculate the gaussian error propagation.',
      url='https://github.com/vale981/SecondaryValue',
      author='Valentin Boettcher',
      author_email='hiro@protagon.space',
      license='GPLv3.0',
      packages=['SecondaryValue'],
      install_requires=[
          'numpy',
          'sympy',
      ],
      zip_safe=True)
