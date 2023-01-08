#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ${SCRIPT_DIR}/../../
sudo docker build -f ${SCRIPT_DIR}/Dockerfile . -t spacetrex/dsrc:latest
