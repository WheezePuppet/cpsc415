#!/usr/bin/env python3
'''
CPSC 415 -- Homework #4 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

import subprocess
import os
import sys
import logging
import pandas as pd


logging.getLogger().setLevel(logging.CRITICAL + 1)

NUM_CORES = 8

if len(sys.argv) != 3:
    print('Usage: score_wumpus.py UMWNetID num_runs.')
    sys.exit(1)
umw_net_id = sys.argv[1]
num_runs = int(sys.argv[2])
runs_per_core = num_runs // NUM_CORES


procs = []
seeds = range(1001,1001+num_runs,runs_per_core)
output_files = [ '/tmp/output'+str(seed)+'.csv' for seed in seeds ]
for seed, output_file in zip(seeds, output_files):
    with open(output_file, 'w') as f:
        procs.append(subprocess.Popen(
            ['./main_wumpus.py',umw_net_id,'suite='+str(runs_per_core),
                                                        str(seed), 'NONE'],
            stdout=f))

print('Waiting for completion...')
[ p.wait() for p in procs ]
print('...done.')

cmd_line = 'cat ' + ' '.join(output_files) 
with open('/tmp/output.csv','w') as f:
    subprocess.Popen(['cat'] + output_files, stdout=f)
os.system('grep -v "^#" /tmp/output.csv > /tmp/output2.csv')
os.system('grep -v "seed" /tmp/output2.csv > /tmp/output3.csv')
os.system('head -1 /tmp/output.csv > /tmp/outputheader.csv')
os.system('cat /tmp/outputheader.csv /tmp/output3.csv > /tmp/output4.csv')
all_of_them = pd.read_csv("/tmp/output4.csv")
with open('/tmp/{}.csv'.format(umw_net_id),'w') as final:
    score_line = '# Score: min {}, max {}, mean {} med {}'.format(
        all_of_them.score.min(), all_of_them.score.max(),
        round(all_of_them.score.mean(),2),
        int(all_of_them.score.median()))
    num_steps_line = '# Num_steps: min {}, max {}, med {}'.format(
        all_of_them.num_steps.min(), all_of_them.num_steps.max(),
        round(all_of_them.num_steps.mean(),2),
        int(all_of_them.num_steps.median()))
    print(score_line)
    print(score_line, file=final)
    print(num_steps_line)
    print(num_steps_line, file=final)
    all_of_them.to_csv(path_or_buf=final,index=False)
os.system('rm /tmp/output*.csv')
print('Output in /tmp/{}.csv.'.format(umw_net_id))

num_lines = int(subprocess.run(['wc', '-l', umw_net_id + '_ExplorerAgent.py'],
    stdout=subprocess.PIPE).stdout.decode('utf-8').split(' ')[0])

os.system('touch wumpus_results.txt')
with open('./wumpus_results.txt','a') as f:
    print('{},{},{},{},{},{},{},{},{}'.format(umw_net_id,
        all_of_them.score.min(), all_of_them.score.max(),
        round(all_of_them.score.mean(),2),
        int(all_of_them.score.median()),
        all_of_them.num_steps.min(), all_of_them.num_steps.max(),
        int(all_of_them.num_steps.median()),
        num_lines), file=f)

