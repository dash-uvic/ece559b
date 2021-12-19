#!/bin/bash

args="-n 100000 -it 0.5 --single" 
args="-n 100000 -it 0.5" 
args="-n 100000 -it 0.70 --single" 

#poster version
bash scripts/run.sh -s boxfinder.py -r qnetv2-googlenet-poster-v4-single-07 -a "$args --mode poster -a qnetv2+googlenet"
bash scripts/run.sh -s boxfinder.py -r qnetv2-resnet50-poster-v4-single-07 -a "$args --mode poster -a qnetv2+resnet50"

#mask version
#bash scripts/run.sh -s boxfinder.py -r qnet-mask-v4-single -a "$args --mode mask --image-size 112"
#bash scripts/run.sh -s boxfinder.py -r qnet-googlenet-mask-v4-single -a "$args --mode mask -a qnet+googlenet"
#bash scripts/run.sh -s boxfinder.py -r qnet-resnet50-mask-v4-single -a "$args --mode mask -a qnet+resnet50"

#draw version
#bash scripts/run.sh -s boxfinder.py -r qnet-draw-v4-single -a "$args --mode draw --image-size 112"
#bash scripts/run.sh -s boxfinder.py -r qnet-googlenet-draw-v4-single -a "$args --mode draw -a qnet+googlenet"
#bash scripts/run.sh -s boxfinder.py -r qnet-resnet50-draw-v4-single -a "$args --mode draw -a qnet+resnet50"
