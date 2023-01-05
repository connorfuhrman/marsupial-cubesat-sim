#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ${SCRIPT_DIR}
sudo apptainer build dsrc.sif docker-daemon://spacetrex/dsrc:latest
