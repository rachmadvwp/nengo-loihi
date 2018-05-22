***********
Nengo Loihi
***********

Setting up Intel server
-----------------------
1. SSH into Intel server.
2. Create your own directory if you haven't already.
3. Create a Python virtualenv for yourself

    mkvirtualenv -p python_nx nxsdk_yourname

4. Install NxSDK. There is a copy in ~/ which I have made a setup.py file for.
  We will keep this up to date as Intel releases new versions. To have control
  over your own NxSDK, clone it to your folder:

    git clone /nfs/ncl/git/NxSDK.git

5. Install NxSDK requirements

    pip install -r NxSDK/requirements.txt

6. Clone and install nengo_loihi

    git clone https://gl.appliedbrainresearch.com/abr/nengo-loihi.git
    cd nengo-loihi
    python setup.py develop


Running on Intel server
-----------------------
1. Activate virtualenv

    workon nxsdk_yourname

2. Run your script on SLURM

    SLURM=1 python yourscript.py
