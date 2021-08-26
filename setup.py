"""
Setup of codebase
Author: Michael Danielczuk
"""
import os
import sys 

from setuptools import setup

requirements = [
    'numpy',
    'torch',
    'tqdm',
    'scikit-image',
    'autolab_core',
    'prettytable',
]

if "--list-reqs" in sys.argv:
    print("\n".join(requirements))
    exit()

# load __version__ without importing anything
version_file = os.path.join(
    os.path.dirname(__file__), "xray/version.py"
)
with open(version_file, "r") as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split("=")[-1])

setup(
    name='xray',
    version = __version__,
    description = 'X-Ray',
    long_description = 'X-Ray Mechanical Search Training and Benchmarking.',
    author = 'Michael Danielczuk',
    author_email = 'mdanielczuk@berkeley.edu',
    license = 'MIT Software License',
    url = 'https://github.com/BerkeleyAutomation/probabilistic-poses',
    keywords = 'robotics grasping computer vision',
    classifiers = [
        'License :: OSI Approved :: MIT Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'
    ],
    packages = ['xray'],
    install_requires = requirements,
)
