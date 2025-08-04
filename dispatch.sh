#!/bin/bash
SLURM_MAX_TASKS=1000  # Could be different- use `scontrol show config | grep MaxArraySize`

send_out_batch() {
    end=$1
    shift
    echo Running batch name=${2} ${1}:0-$end
    sbatch --job-name=${2} --dependency=singleton --array=0-$end run_batch.sh $jobs_run $@
}

if [[ -z $1 ]]; then
    echo You must pass a batch name. I\'m not smart enough to make my own!
    exit 1
fi
batchname=$1
shift

if [[ -z $1 ]]; then
    batch_dirs=(data/sweep_configs/*/configs.csv)
else
    batch_dirs=$@
fi

full_file=$(mktemp)
cat "${batch_dirs[@]}" > $full_file

lines_left=$(wc -l < $full_file)
jobs_run=0
while [[ $lines_left -gt $SLURM_MAX_TASKS ]]; do
    send_out_batch $SLURM_MAX_TASKS $full_file $batchname
    jobs_run=$(( $jobs_run + $SLURM_MAX_TASKS ))
    lines_left=$(( $lines_left - $SLURM_MAX_TASKS ))
done

if [[ $lines_left -gt 0 ]]; then
    send_out_batch $lines_left $full_file $batchname
fi

rm $full_file
