
from setuptools import setup
import os, sys
sys.path.append(os.path.dirname(__file__))

setup(
    name='netsanut',
    version=0.1,
    packages=[
        'netsanut',
    ],
    author='Weijiang Xiong',
    author_email='weijiangxiong1998@gmail.com',
    install_requires=[
    "torch",
    "torchvision",
    "matplotlib",
    "numpy",
    ],
)