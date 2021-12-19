#!/usr/bin/env python3

'''
This script exctracts training variables from all logs from 
tensorflow event files ("event*"), writes them to Pandas 
and finally stores in long-format to a CSV-file including
all (readable) runs of the logging directory.

The magic "5" infers there are only the following v.tags:
[lr, loss, acc, val_loss, val_acc]
'''

#import tensorflow as tf
import glob
import os
import pandas as pd
import argparse
from tensorboard.backend.event_processing import event_file_loader


def extract_runlog(path):
    runlog = pd.DataFrame(columns=['metric', 'value'])
    metrics=[]
    try:
        loader = event_file_loader.EventFileLoader(path)
        for e in list(loader.Load()):
            for v in e.summary.value:
                r = {'metric': v.tag, 'value':v.tensor.float_val[0]}
                metrics.append(v.tag)
                runlog = runlog.append(r, ignore_index=True)
    # Dirty catch of DataLossError
    except Exception as e:
        print(e)
        #print('Event file possibly corrupt: {}'.format(path))
        return None

    metrics = set(metrics)
    N = len(metrics)
    if N == 0:
        return None 
    print(f"metrics={metrics}  N={N}")

    runlog['epoch'] = [item for sublist in [[i]*N for i in range(0, len(runlog)//N)] for item in sublist]
    
    return runlog

def convert_to_pandas(event_files):
    # Call & append
    all_log = pd.DataFrame()
    for path in event_files:
        print(f"| parsing {path}")
        log = extract_runlog(path)
        if log is not None:
            if all_log.shape[0] == 0:
                all_log = log
            else:
                all_log = all_log.append(log)

    # Inspect
    print(all_log.shape)
    all_log.head()    
                
    # Store
    all_log.to_csv(f'{args.folder}.csv', index=None)

if __name__ == "__main__":
    # Get all event* runs from logging_dir subdirectories
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    #parser.add_argument('--tag', default='test_metric')
    args = parser.parse_args()

    logging_dir = f'{args.folder}/logs'
    event_paths = glob.glob(os.path.join(logging_dir, "event*"))
    convert_to_pandas(event_paths)
