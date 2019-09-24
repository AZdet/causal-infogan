from setuptools import setup
import numpy

setup(
    name='CIGAN',
    version='0.2dev',
    packages=['vpa'],
    license='MIT License',
    include_dirs=[numpy.get_include(),],
)