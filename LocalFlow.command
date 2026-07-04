#!/bin/zsh
# Double-click to start localflow. First run creates the environment.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    echo "first run — installing dependencies ..."
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt || exit 1
fi
exec .venv/bin/python localflow.py
