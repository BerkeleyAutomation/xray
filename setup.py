"""
Setup of codebase
Author: Michael Danielczuk
"""
from setuptools import setup

requirements = [
    'numpy',
    'torch',
    'tqdm',
    'scikit-image',
    'autolab_core',
    'apex'
]

exec(open('xray/version.py').read())

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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'
    ],
    packages = ['xray'],
    install_requires = requirements,
)
