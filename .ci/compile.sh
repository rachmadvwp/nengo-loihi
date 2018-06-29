#!/usr/bin/env bash

NAME=$0
COMMAND=$1
NLDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

function usage {
    echo "usage: $NAME <command>"
    echo
    echo "  setup   Set up ssh agent and git configuration"
    echo "  run     Compile Python to object files"
    echo "  deploy  Commit compiled files and push to remote"
    exit 1
}

if [[ "$COMMAND" == "setup" ]]; then
    eval $(ssh-agent -s)
    git config --global user.name "Applied Brain Research"
    git config --global user.email "info@appliedbrainresearch.com"
    git clone ssh://git@gl.appliedbrainresearch.com:36563/abr/nengo-loihi-compiled.git
elif [[ "$COMMAND" == "run" ]]; then
    python .ci/compile.py
elif [[ "$COMMAND" == "deploy" ]]; then
    NLCOMMIT=$(git rev-parse --short HEAD)
    cd nengo-loihi-compiled
    git add -A
    git commit --allow-empty -m "Compiled nengo-loihi at $NLCOMMIT"
    git push origin master
else
    if [[ -z "$COMMAND" ]]; then
        echo "Command required"
    else
        echo "Command $COMMAND not recognized"
    fi
    echo
    usage
fi
