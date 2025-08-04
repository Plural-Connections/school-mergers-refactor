#!/bin/bash
SLURM_MAX_TASKS=1000  # Could be different- use `scontrol show config | grep MaxArraySize`

if [[ -z $1 ]]; then
    echo You must pass a batch name. I\'m not smart enough to make my own!
    exit 1
fi
batchname=$1
shift

if [[ -z $2 ]]; then
    batch_dirs=(data/sweep_configs/*)
else
    batch_dirs=()
    for i in "$@"; do
        batch_dirs+=("data/sweep_configs/$i")
    done
fi

full_file=$(mktemp)
echo Temporary file is $full_file
cat "${batch_dirs[@]}" > $full_file

lines_left=$(wc -l < $full_file)
jobs_run=0
while [[ $lines_left -gt $SLURM_MAX_TASKS ]]; do
    echo sbatch --array=${jobs_run}-$(( $jobs_run + $SLURM_MAX_TASKS )) run_batch.sh $full_file $batchname
    jobs_run=$(( $jobs_run + $SLURM_MAX_TASKS ))
    lines_left=$(( $lines_left - $SLURM_MAX_TASKS ))
done

if [[ $lines_left -gt 0 ]]; then
    echo sbatch --array=${jobs_run}-$(( $jobs_run + $lines_left )) run_batch.sh $full_file $batchname
fi

echo Done!
