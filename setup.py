
from setuptools import setup
import os, sys
sys.path.append(os.path.dirname(__file__))

setup(
    name='netsanut',
    version=0.3,
    packages=[
        'netsanut',
    ],
    author='Weijiang Xiong',
    author_email='weijiangxiong1998@gmail.com',
    install_requires=[
    "torch",
    "torch_geometric", # for graph neural networks
    "torchvision",
    "matplotlib",
    "numpy",
    "seaborn", # for better color in plots
    "einops", # for rearranging tensors 
    "omegaconf", # configuration files
    "tabulate", # for pretty print
    "termcolor", # for colored print in terminal
    "tqdm", # for progress bar
    "scipy", # for scientific computing
    ],
    # optional dependencies
    extras_require={
        "vis": [ # for visualization
        "geopandas", # for geospatial data
        "contextily", # for basemap
        ]
    },
)