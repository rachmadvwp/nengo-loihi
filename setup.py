#!/usr/bin/env python
import imp
import io
import os
import sys

from setuptools import find_packages, setup


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    "version", os.path.join(root, "nengo_loihi", "version.py"))
testing = "test" in sys.argv or "pytest" in sys.argv

setup(
    name="nengo_loihi",
    version=version_module.version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    scripts=[],
    url="https://github.com/nengo/nengo-loihi",
    license="Free for non-commercial use",
    description="Run Nengo models on the Loihi chip",
    long_description=read("README.rst"),
    zip_safe=False,
    setup_requires=[
        "nengo",
    ],
    install_requires=[
        "nengo",
    ],
    tests_require=[
        "pytest>=3.2",
    ],
    entry_points={
        'nengo.backends': [
            'loihi = nengo_loihi:Simulator'
        ],
    },
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
