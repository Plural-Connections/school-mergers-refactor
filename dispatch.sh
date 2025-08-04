#!/bin/bash
SLURM_MAX_TASKS=1000  # Could be different- use `scontrol show config | grep MaxArraySize`

send_out_batch() {
    echo Running batch name=${4} ${3}:${1}-${2}
    sbatch --job-name=${4} --dependency=singleton --array=${1}-${2} run_batch.sh ${3} ${4}
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
echo Temporary file is $full_file
cat "${batch_dirs[@]}" > $full_file

lines_left=$(wc -l < $full_file)
jobs_run=0
while [[ $lines_left -gt $SLURM_MAX_TASKS ]]; do
    send_out_batch ${jobs_run} $(( $jobs_run + $SLURM_MAX_TASKS - 1 )) $full_file $batchname
    jobs_run=$(( $jobs_run + $SLURM_MAX_TASKS ))
    lines_left=$(( $lines_left - $SLURM_MAX_TASKS ))
done

if [[ $lines_left -gt 0 ]]; then
    send_out_batch ${jobs_run} $(( $jobs_run + $lines_left - 1 )) $full_file $batchname
fi

echo Done!
