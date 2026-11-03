#!/bin/bash
# start the real-data queue once all 17 datasets are preprocessed
until [ "$(ls /home/ubuntu/data/robot_video/prep/*/meta.json 2>/dev/null | wc -l)" -ge 17 ]; do sleep 30; done
sleep 20
exec ./run_real.sh
