from setuptools import setup
from setuptools import find_packages


setup(name='rnnfwk',
      version='0.1',
      description='Recurrent Neural Network Framework',
      author='Joerg Franke',
      author_email='joergfranke@zoho.com',
      url='https://github.com/joergfranke/rnnfwk',
      license='MIT License',
      install_requires=['theano>=0.8.2'],
      packages=find_packages())