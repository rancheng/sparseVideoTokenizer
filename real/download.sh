#!/bin/bash
# meta + action parquet + the first video file of one camera per dataset
mkdir -p /home/ubuntu/data/robot_video/raw; cd /home/ubuntu/data/robot_video
while read d cam kind; do
  hf download lerobot/$d --type dataset --local-dir raw/$d --include "meta/*" --include "data/*" --include "videos/$cam/chunk-000/file-000.mp4" > /dev/null 2>&1 && echo "ok $d $cam" || echo "FAIL $d $cam"
done < /home/ubuntu/Codes/videoTokenizer/real/datasets.txt
echo DL_DONE
