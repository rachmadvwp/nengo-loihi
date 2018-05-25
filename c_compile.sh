#!/usr/bin/env bash

SOURCE_DIR=nengo_loihi
TARGET_DIR=nengo_loihi_c

if [[ $1 != "" ]];
then
    TARGET_DIR="$1"
fi

echo "TARGET_DIR = $TARGET_DIR"

python setup_c.py build_ext --inplace

rsync -avm --exclude-from=c_compile_excludes.txt "$SOURCE_DIR/" $TARGET_DIR
rsync -av "$SOURCE_DIR/__init__.py" $TARGET_DIR
