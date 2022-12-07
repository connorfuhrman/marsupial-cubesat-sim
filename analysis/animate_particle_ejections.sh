#!/bin/bash

# Script to animate particle ejections.
#
# First uses GNUplot to generate indivdidual
# frames to a png at /tmp/bennu_particle_ejection_frames/frame_<N>.png
# then uses ffmpeg to generate a .mp4 video file. 
#
# Note: This will take some time!
#
# TODO parallelize GNUplot portion

set -e


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ${SCRIPT_DIR}/../bennu_particles/generated

gnuplot ${SCRIPT_DIR}/animate_particles.gnuplot

ffmpeg -framerate 30 -pattern_type glob -i '/tmp/bennu_particle_ejection_frames/*.png' \
       -c:v libx264 -pix_fmt yuv420p ${SCRIPT_DIR}/particle_ejection_simulation.mp4
    
