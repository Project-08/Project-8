#!/bin/bash

docker container run\
  --rm\
  --volume $(pwd):/docker/src/\
  --tty\
  --interactive\
  ${CONTAINER:-project8} \
  bash -c\
  "trap 'sleep 1; exit' SIGINT\
  && cd /docker/src\
  && $*"
