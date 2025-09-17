#! -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages
import re
from typing import List

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()


def get_version() -> str:
    with open(os.path.join("bert4torch", "cli.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("VERSION")
        version = re.findall(pattern, file_content)[0]
        return version


def get_requires() -> List[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


def get_console_scripts() -> List[str]:
    console_scripts = [
        "bert4torch = bert4torch.cli:main",
        "b4t = bert4torch.cli:main"
        ]
    return console_scripts


extra_require = {
    "transformers": ["transformers"],
    "accelerate": ["accelerate"],
    "deepspeed": ["deepspeed>=0.10.0,<=0.16.5"],
    "trl": ["trl"],
    "peft": ["peft"]
}


setup(
    name='bert4torch',
    version=get_version(),
    author="Tongjilibo",
    author_email="tongjilibo@163.com",
    description='an elegant bert4torch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT Licence',
    url='https://github.com/Tongjilibo/bert4torch',
    install_requires=get_requires(),
    extras_require=extra_require,
    packages=find_packages(),
    entry_points={"console_scripts": get_console_scripts()},
)