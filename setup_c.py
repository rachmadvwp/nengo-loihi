#!/usr/bin/env python
import imp
import io
import os
import sys

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py")

from distutils.extension import Extension
from Cython.Distutils import build_ext

root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, 'nengo_loihi', 'version.py'))
testing = 'test' in sys.argv or 'pytest' in sys.argv


def allpy(path):
    for obj in os.listdir(path):
        objpath = os.path.join(path, obj)
        if os.path.isdir(objpath):
            for pypath in allpy(objpath):
                yield pypath
        else:
            if os.path.splitext(objpath)[1] == '.py':
                yield objpath


ext_modules = []
for pypath in allpy('nengo_loihi'):
    modname = os.path.splitext(pypath)[0].replace(os.sep, '.')
    ext_modules.append(Extension(modname, [pypath]))


setup(
    name="nengo_loihi",
    version=version_module.version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    scripts=[],
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
