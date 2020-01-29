import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="confbusterplusplus",
    version="1.0.0",
    author="Eric Dang",
    description=("A macrocycle conformational sampling tool."),
    license="MIT",
    keywords="macrocycle conformer sampling",
    url="https://github.com/e-dang/ConfBusterPlusPlus.git",
    packages=find_packages(exclude=('examples')),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6"
    ],
)
