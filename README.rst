***********
Nengo Loihi
***********

Running on the Intel server
===========================

``conda`` setup
---------------

Currently, the easiest way to run models
on the Intel server requires Conda and Python 3.5.5.
The following commands will install Miniconda to ``~/miniconda``
and set up a Loihi environment::

  wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p "$HOME/miniconda"
  export PATH="$HOME/miniconda/bin:$PATH"
  conda create --name loihi python=3.5.5
  source activate loihi
  conda install cython mkl numpy
  cd path/to/nengo-loihi
  pip install -e .

If you want to run models locally with the simulator,
then you should also do::

  conda install matplotlib scipy

Note that the ``PATH`` will only be changed
for the current terminal session.
To add it permanently, modify ``.bash_profile``.

``ssh`` setup
-------------

Create ``$HOME/.ssh/config`` if it does not yet exist,
then add the following to it::

  Host intelhost
       User celiasmith
       HostName ncl-abr.research.intel-research.net
       ProxyCommand ssh celiasmith@ssh.intel-research.net -W %h:%p

This enables you to do ``ssh intelhost`` to log into the Intel server.

In a terminal, do the following commands::

  ssh-keygen
  # Hit enter to go through all the prompts, no need to add a passphrase
  ssh-copy-id intelhost

``ssh-copy-id`` will ask you for a password.
Once you provide it, your SSH key will be copied
to the Intel server so that
you can now log in without a password.

Running a model
---------------

``nengo_loihi`` includes a ``loihi`` command.
The easiest way to run a model on Intel's server is::

  loihi run <model.py> intelhost <your folder> <output files>

For example, if you have a script called ``oscillator.py``
which creates two figures,
``oscillator-01.png`` and ``oscillator-02.png``,
running::

  loihi run oscillator.py intelhost my-folder oscillator-01.png oscillator-02.png

will run that example and download the output figures
to the current folder.

For more commands and other help, run::

  loihi --help

Or::

  loihi <command> --help

Under the hood
--------------

The ``loihi run`` command does the following.

1. Compile the files in ``nengo_loihi`` to ``.so`` object files
   in order to keep the source code off of the Intel server.

2. Upload those files to the specified ``DST`` folder.
   You will end up with a ``DST/nengo_loihi`` folder
   with ``.so`` files and ``__init__.py``.

3. Upload the model to ``DST/$(basename MODEL)``.

4. Run the model with::

     cd DST
     workon <value of --env>
     SLURM=1 python $(basename MODEL)

5. Download each ``OUTPUT`` file with ``scp``.

6. If you did not pass ``--no-clean``,
   ``MODEL`` and ``OUTPUT`` files are removed from the server.
   The ``nengo_loihi`` files remain to make future uploads faster.

Advanced
--------

Most people will not have to follow these steps.
However, if you are modifying ``NxSDK`` files,
or doing something else advanced,
you may want to do the following.

First, set up your own Python virtualenv on the Intel server.
SSH into the Intel server, then::

  mkvirtualenv -p python_nx <environment name>
  mkdir <your folder>
  cd <your folder>
  git clone "$HOME/NxSDK"
  pip install -r NxSDK/requirements.txt
  pip install -e NxSDK

Then, when running a file with ``loihi run``,
provide your environment name
with the ``--env`` flag::

  loihi run oscillator.py intelhost my-folder oscillator-01.png oscillator-02.png --env nxsdk_me

You can also pass in a different name for the server
if you have already set it up under a different name.
Or, you can pass in any host string that SSH understands::

  loihi run oscillator.py me@192.168.0.1 my-folder

See ``loihi --help`` or ``nengo_loihi/__main__.py`` for more details.
