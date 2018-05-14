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


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, 'nengo_loihi', 'version.py'))
testing = 'test' in sys.argv or 'pytest' in sys.argv

ext_modules = []
if 1:
    from distutils.extension import Extension
    from Cython.Distutils import build_ext

    def allpy(path):
        for obj in os.listdir(path):
            objpath = os.path.join(path, obj)
            if os.path.isdir(objpath):
                for pypath in allpy(objpath):
                    yield pypath
            else:
                if os.path.splitext(objpath)[1] == '.py':
                    yield objpath

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
    # url="https://github.com/nengo/nengo_loihi",
    license="Free for non-commercial use",
    description="Run Nengo models on the Loihi chip",
    long_description=read('README.rst'),
    zip_safe=False,
    setup_requires=[
        "nengo",
    ],
    install_requires=[
        "nengo",
    ],
    tests_require=[
        'pytest>=3.2',
    ],
    # entry_points={
    #     'nengo.backends': [
    #         'reference = nengo:Simulator'
    #     ],
    # },
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
