#!/bin/bash

. scripts/utils.sh

commit=$(git hash)

echo "Using state file: $state_file"

module load python/3.7
source $HOME/py37/bin/activate

echo "${BASH_SOURCE[0]}:$commit $args" > $output_path/currently_running.txt
git diff > $output_path/git-patch.diff 

echo "python $script --data-dir /home/adash/data/oxford-III-pets --resume --state-file $state_file $args"
python $script --data-dir /home/adash/data/oxford-III-pets --resume --state-file $state_file $args 

error_code=$?
echo "Program finished with error code: $error_code"

# Resubmit if not all work has been done yet.
# You must define the function work_should_continue().
if work_should_continue; then
     if [ $error_code -eq 2 ]; then
            echo "Distributed setup must have timedout.  Return to queue: $state_file"
     #elif [ $error_code -eq 1 ];then
     #    echo "Distributed setup must have had a CUDA error. Return to queue: $state_file"
     elif [ $error_code -ne 0 ]; then
            echo "Cancelling job."
	    scancel $SLURM_ARRAY_JOB_ID
	    mv $state_file ${state_file}.error
     else
     	echo "Still have work to do on ${BASH_SOURCE[0]}:$state_file ... "
     fi 
else
     show_time
     if [ -z "$no_save" ]; then
         echo "Moving $output_dir to $HOME/results"
         rsync -aRt  $output_path $HOME/results
     else
         echo "WARNING: Not backing up ${output_path} to $HOME/results, will be deleted in 60 days"
     fi

     if [ ! -z "$SLURM_ARRAY_TASK_COUNT" ];then
     	scancel $SLURM_ARRAY_JOB_ID
     fi
fi

