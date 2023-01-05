#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm ${SCRIPT_DIR}/apptainer/*.sif

bash ${SCRIPT_DIR}/docker/build.sh
bash ${SCRIPT_DIR}/apptainer/build_from_local_docker.sh
