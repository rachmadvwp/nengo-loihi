***********
Nengo Loihi
***********

Setting up Intel server
-----------------------

1. SSH into Intel server.
2. Create your own directory if you haven't already.
3. Create a Python virtualenv for yourself::

      mkvirtualenv -p python_nx nxsdk_yourname

4. Install NxSDK. There is a copy in ~/ which I have made a setup.py file for.
   We will keep this up to date as Intel releases new versions. To have control
   over your own NxSDK, clone it to your folder::

      git clone /nfs/ncl/git/NxSDK.git

5. Install NxSDK requirements::

      pip install -r NxSDK/requirements.txt

6. Install and compile nengo_loihi on your own machine
   (to keep source off Intel server)::

      git clone https://gl.appliedbrainresearch.com/abr/nengo-loihi.git
      cd nengo-loihi
      ./c_compile.sh <target_dir>

   where <target_dir> defaults to "nengo_loihi_c" if not specified. To run on
   the Intel server, you must compile with Python 3.5.5 with the ``with-fpectl``
   flag unset. The easiest way I've found to do this is install Miniconda
   and make a new conda environment with Python==3.5.5::

      conda create --name py35 python=3.5.5
      source activate py35

Running on Intel server
-----------------------

1. Activate virtualenv::

      workon nxsdk_yourname

2. Put the compiled nengo_loihi directory in the same folder as your target
   script (must be named "nengo_loihi").

2. Run your script on SLURM::

      SLURM=1 python yourscript.py
