#!/bin/bash
Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &
export DOCKER_HOST_IP=$1
bash
# cd /usr/src/app/src && python test.py
# cd /usr/src/app/src && ./stream.py -i ../videos/90CC.mp4 -r 90 -k 0 -o ../output/90CC.mp4
