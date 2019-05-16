from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
def readme():
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        return f.read()

setup(name='SecondaryValue',
      version='0.1.7',
      description='A helper to calculate the gaussian error propagation.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/vale981/SecondaryValue',
      author='Valentin Boettcher',
      author_email='hiro@protagon.space',
      license='GPLv3.0',
      packages=['SecondaryValue'],
      keywords='gaussian error propagation',
      install_requires=[
          'numpy',
          'sympy',
      ],
      zip_safe=True)
