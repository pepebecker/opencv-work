#!/bin/bash
Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &
export DOCKER_HOST_IP=$1
bash
