#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='bert4torch',
    version='0.1.7',
    description='an elegant bert4torch',
    long_description='bert4torch: https://github.com/Tongjilibo/bert4torch',
    license='MIT Licence',
    url='https://github.com/Tongjilibo/bert4torch',
    author='Tongjilibo',
    install_requires=['torch>1.0'],
    packages=find_packages()
)