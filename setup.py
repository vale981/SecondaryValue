from setuptools import setup


def readme():
    with open('Readme.md') as f:
        return f.read()

setup(name='SecondaryValue',
      version='0.0.9',
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
