#from setuptools import setup
#from setuptools import find_packages
from distutils.core import setup


setup(name='recnet',
      version='0.1',
      description='recnet - Recurrent Neural Network Framework',
      author='Joerg Franke',
      author_email='joergfranke@zoho.com',
      url='https://github.com/joergfranke/recnet',
      license='MIT License',
      install_requires=['theano>=0.8.2', 'klepto==0.1.2', 'numpy>=1.11.1'],
      packages=['recnet',])

