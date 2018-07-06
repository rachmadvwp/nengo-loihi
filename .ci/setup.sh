#!/usr/bin/env bash

conda config --set always_yes yes --set changeps1 no

# Exit if the loihi env already exists
if conda info --envs | grep loihi > /dev/null; then
    return 0
fi

# Otherwise, set it up
conda update --quiet conda
conda create --quiet --name loihi python=3.5.5
source activate loihi
conda install --quiet cython matplotlib mkl numpy scipy
pip install flake8 jinja2 nengo pylint pytest
source deactivate
