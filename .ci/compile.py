import os
import shutil
from setuptools import setup

from Cython.Build import cythonize

d_root = os.path.realpath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
d_tmp = "build-tmp"
d_build = os.path.join(d_root, "nengo-loihi-compiled")
d_loihi = os.path.join(d_build, "nengo_loihi")


def compile_nengo_loihi():
    """Compile .py files to .so with Cython."""

    # Remember current directory
    prevdir = os.getcwd()

    # Switch to nengo-loihi root
    os.chdir(d_root)

    # Compile to .so with Cython and distutils
    setup(
        name="nengo_loihi",
        ext_modules=cythonize(
            "nengo_loihi/**/*.py",
            exclude=[
                "**/__init__.py",
                "**/conftest.py",
                "**/tests/**",
            ],
            build_dir=d_tmp,
        ),
        script_name="setup.py",
        script_args=[
            "build_ext",
            "--quiet",
            "--build-lib", d_build,
            "--build-temp", ".",
        ],
    )

    # Switch back to the previous directory (user might care)
    os.chdir(prevdir)

    # Also copy __init__.py to the build directory
    shutil.copyfile(os.path.join(d_root, "nengo_loihi", "__init__.py"),
                    os.path.join(d_loihi, "__init__.py"))


if __name__ == "__main__":
    compile_nengo_loihi()
