#!/bin/bash

start_time=$(date +%s)
state_file=${start_time}.pt
curr_date=$(date +"%Y-%m-%d")

#Defaults
gpu=1
node=1
train_time="5:00"
script="boxfinder.py"
no_save=""
gtype="p100"

while [[ "$#" -gt 0 ]]; do case $1 in
  -r|--run) run_name="$2"; shift;;
  -a|--args) args="$2"; shift;;
  -t|--train) train_time="$2"; shift;;
  -ns|--no-save) no_save="true";; 
  -s|--script) script="$2"; shift;;
  -d|--date) curr_date="$2"; shift;;
  -h|--help) echo "USAGE: $0 -r <run_name> -g [ngpu|$gpu] -n [node|$node] [-a '<args>']"; exit 1;;  
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

if [ -z "$run_name" ]; then
    echo "-r,--run [name] is required."
    exit
fi

server=$(hostname)
echo "Using gpu: ${gtype}"

#Cedar: no more than 8 cpus per v100l
#Graham: no more than 16 cpus
output_path="${curr_date}/${run_name}"
state_file="${output_path}/${state_file}"
train_args="--output-dir $output_path $args"

echo "Creating directory $output_path/logs and state file $state_file"
mkdir -p ${output_path}/logs 
touch $state_file 

echo "Adding to queue: ${run_name}"
echo "args: ${train_args}"

sbatch --account=def-branzana \
       --nodes=$node \
       --gres=gpu:${gtype}:${gpu} \
       --ntasks-per-node=32 \
       --mem=32G \
       --time=0-$train_time \
       --array=1-5%1 \
       --job-name ${run_name} \
       --output=$output_path/logs/log-%N-%j.out \
       --export=ALL,args="$train_args",output_path="$output_path",start_time="$start_time",state_file="$state_file",script="$script",no_save="$no_save" \
       scripts/train.sh
