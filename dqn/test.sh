#!/bin/bash
args="--output-dir deleteme -n 10 --save-freq 2 -it 0.5"
#args="--output-dir deleteme --batch-size 8 -n 10 --save-freq 2 -it 0.5 --max-steps 50"
args="--output-dir deleteme --batch-size 8 -n 10 --save-freq 2 -it 0.5 --max-steps 50 --prioritized"

test="qnetv2+resnet50"
python boxfinder.py $args --mode poster -a qnetv2+resnet50 
if [[ $? -ne 0 ]]; then
    echo "Failed ${test}"
    exit
fi
exit
test="mask_googlenet"
python boxfinder.py  $args  --mode poster -a qnetv2+googlenet
if [[ $? -ne 0 ]]; then
    echo "Failed ${test}"
    exit
fi

test="mask_qnet"
python boxfinder.py  $args  --mode mask -a qnet --image-size 112
if [[ $? -ne 0 ]]; then
    echo "Failed ${test}"
    exit
fi

test="mask_resnet50"
python boxfinder.py  $args  --mode mask -a qnet+resnet50 
if [[ $? -ne 0 ]]; then
    echo "Failed ${test}"
    exit
fi

test="mask_googlenet"
python boxfinder.py  $args --mode mask -a qnet+googlenet
if [[ $? -ne 0 ]]; then
    echo "Failed ${test}"
    exit
fi


test="draw_qnet"
python boxfinder.py $args  --mode draw -a qnet --image-size 112
if [[ $? -ne 0 ]]; then
    echo "Failed ${test}"
    exit
fi

test="draw_resnet50"
python boxfinder.py $args  --mode draw -a qnet+resnet50
if [[ $? -ne 0 ]]; then
    echo "Failed ${test}"
    exit
fi

test="draw_googlenet"
python boxfinder.py $args --mode draw -a qnet+googlenet
if [[ $? -ne 0 ]]; then
    echo "Failed ${test}"
    exit
fi
