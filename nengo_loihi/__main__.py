import errno
import os
import shutil
import subprocess
from setuptools import setup

import click
from Cython.Build import cythonize


class Context(object):
    def __init__(self):
        self.root = os.path.realpath(
            os.path.join(os.path.dirname(__file__), os.pardir)
        )
        self.tmp = "build-tmp"

    @property
    def build_dir(self):
        return os.path.join(self.root, "build")

    @property
    def loihi_dir(self):
        return os.path.join(self.build_dir, "nengo_loihi")


def info(msg):
    click.secho(msg, fg="blue", bold=True, underline=True)


def error(msg):
    click.secho(msg, fg="red", bold=True)


@click.group()
@click.pass_context
def main(ctx):
    """Utilities for running models on remote Loihis.

    This command assumes that you have set up conda and SSH
    as described in README.rst.
    """
    ctx.obj = Context()


@main.command()
@click.pass_obj
def compile(obj):
    """Compile .py files to .so with Cython."""

    # Remember current directory
    prevdir = os.getcwd()

    # Switch to nengo-loihi root
    os.chdir(obj.root)

    # Compile to .so with Cython and distutils
    info("Compiling .so files")
    setup(
        name="nengo_loihi",
        ext_modules=cythonize(
            "nengo_loihi/**/*.py",
            exclude=[
                "**/__init__.py",
                "**/__main__.py",
                "**/conftest.py",
                "**/tests/**",
            ],
            build_dir=obj.tmp,
        ),
        script_name="setup.py",
        script_args=[
            "build_ext",
            "--quiet",
            "--build-lib", obj.build_dir,
            "--build-temp", ".",
        ],
    )

    # Switch back to the previous directory (user might care)
    os.chdir(prevdir)

    # Build .so files should be in build/nengo_loihi
    assert os.path.exists(os.path.join(obj.build_dir, "nengo_loihi"))

    # Also copy __init__.py to the build directory
    shutil.copyfile(os.path.join(obj.root, "nengo_loihi", "__init__.py"),
                    os.path.join(obj.loihi_dir, "__init__.py"))
    return 0


@main.command()
@click.argument("dst", type=str)
@click.argument("host", type=str, default="localhost")
@click.pass_context
def sync(ctx, dst, host="localhost"):
    """Sync .so files to the DST folder on HOST."""
    ret = ctx.invoke(compile)
    if ret != 0:
        return ret

    info("Syncing nengo_loihi files")

    if host == "localhost":
        try:
            os.makedirs(dst)
        except OSError as err:
            if err.errno == errno.EEXIST and os.path.isdir(dst):
                pass
            else:
                raise
    else:
        ret = subprocess.call([
            "ssh", "-t", host, "mkdir -p {}".format(dst),
        ])
        if ret != 0:
            error("ssh command failed!")
            return ret

    ret = subprocess.call([
        "rsync", "-avm", ctx.obj.loihi_dir,
        dst if host == "localhost" else "{}:{}".format(host, dst),
    ])
    if ret != 0:
        error("rsync command failed!")
        return ret
    return 0


@main.command()
@click.argument("model", type=click.Path())
@click.argument("host", type=str)
@click.argument("dst", type=str)
@click.argument("outputs", type=click.Path(), nargs=-1)
@click.option("--env", default="nxsdk", type=str,
              help="Virtualenv to activate on board")
@click.option("--clean/--no-clean", default=True,
              help="Remove model and plots after running?")
@click.pass_context
def run(ctx, model, host, dst, outputs, env, clean):
    """Run MODEL on Intel's Loihi.

    \b
    MODEL     : Python script with the model to run.
    HOST      : SSH host.
    DST       : Destination folder on host.
    [OUTPUTS] : Files to download after the remote run finishes
    """
    modelbase = os.path.basename(model)
    ret = ctx.invoke(sync, dst=dst, host=host)
    if ret != 0:
        return ret

    info("Uploading {}".format(modelbase))
    ret = subprocess.call([
        "scp", model, "{}:{}/{}".format(host, dst, modelbase)
    ])
    if ret != 0:
        error("scp command failed!")
        return ret

    info("Running {}".format(modelbase))
    ret = subprocess.call([
        "ssh", "-t", host,
        "cd {} && workon {} && SLURM=1 python {}".format(dst, env, modelbase),
    ])
    if ret != 0:
        error("ssh command failed!")
        return ret

    if clean:
        info("Cleaning up {}".format(modelbase))
        ret = subprocess.call([
            "ssh", "-t", host, "rm -f {}/{}".format(dst, modelbase)
        ])
        if ret != 0:
            error("ssh command failed!")
            return ret

    for output in outputs:
        info("Downloading {}".format(output))
        ret = subprocess.call([
            "scp", "{}:{}/{}".format(host, dst, output), output
        ])
        if ret != 0:
            error("scp command failed!")
            return ret

        if clean:
            info("Cleaning up {}".format(output))
            ret = subprocess.call([
                "ssh", "-t", host, "rm -f {}/{}".format(dst, output)
            ])
            if ret != 0:
                error("ssh command failed!")
                return ret

    return 0


if __name__ == "__main__":
    main()
