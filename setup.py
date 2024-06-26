
from setuptools import setup
import os, sys
sys.path.append(os.path.dirname(__file__))

setup(
    name='netsanut',
    version=0.2,
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
    "pandas==2.2.0", # dataset processing
    "pandarallel", # parallel processing for pandas
    "seaborn", # for better color in plots
    "einops", # for rearranging tensors 
    "omegaconf", # configuration files
    "termcolor", # for colored print in terminal
    "geopandas", # for geospatial data
    "contextily", # for basemap
    ],
)