#!/bin/bash
set -eu

USER=$(whoami)
GROUP_NAME="external-sort"
PROGRAM="build/main"

if ! cgget -r memory.limit_in_bytes -n $GROUP_NAME; then
  sudo cgcreate -t "$USER:$USER" -a "$USER:$USER" -g memory:$GROUP_NAME
  cgset -r memory.limit_in_bytes=1M $GROUP_NAME
fi

if [[ $# -eq 1 ]]; then
  cgset -r memory.limit_in_bytes="$1" $GROUP_NAME
  cgget -r memory.limit_in_bytes -n $GROUP_NAME
fi

cgexec -g memory:$GROUP_NAME -- $PROGRAM
