#!/bin/bash
args="--output-dir deleteme --batch-size 8 --start_steps 5 --save-freq 2 -it 0.5 --max-steps 1000 --single"

test="qnetv2+resnet50"
python main.py $args --mode poster 
if [[ $? -ne 0 ]]; then
    echo "Failed ${test}"
    exit
fi
exit
