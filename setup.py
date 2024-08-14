#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='bert4torch',
    version='v0.5.3',
    description='an elegant bert4torch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT Licence',
    url='https://github.com/Tongjilibo/bert4torch',
    author='Tongjilibo',
    install_requires=['numpy', 'tqdm', 'torch>1.6', 'torch4keras==0.2.6', 'six'],
    packages=find_packages(),
    entry_points={"console_scripts": ["bert4torch-llm-server = bert4torch.pipelines.chat:main"]},

)