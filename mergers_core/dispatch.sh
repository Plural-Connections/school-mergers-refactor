#!/bin/bash -e
SLURM_MAX_TASKS=1000  # Could be different- use `scontrol show config | grep MaxArraySize`

__cleanup() {
    scancel -u $USER
    skill -u $USER
}

send_out_batch() {
    array_end=$1; shift
    command="sbatch --job-name=${2} --array=0-$array_end run_batch.sh $jobs_run $@"
    echo $command
    $command
}

trap __cleanup INT HUP TERM

if [[ -z $1 ]]; then
    echo You must pass a batch name. I\'m not smart enough to make my own!
    exit 1
fi

full_file=data/sweep_configs/configs.csv

lines_left=$(wc -l < $full_file)
jobs_run=0
while [[ $lines_left -gt $SLURM_MAX_TASKS ]]; do
    send_out_batch $SLURM_MAX_TASKS $full_file
    jobs_run=$(( $jobs_run + $SLURM_MAX_TASKS ))
    lines_left=$(( $lines_left - $SLURM_MAX_TASKS ))
done

if [[ $lines_left -gt 0 ]]; then
    send_out_batch $lines_left $full_file
fi

echo make sure to remove $full_file when you\'re done!
