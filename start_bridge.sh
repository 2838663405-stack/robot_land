#!/bin/bash
export HOME=/root
source /root/legged-robot/install/setup.bash
exec python3 /root/ws_ros_bridge.py
