#!/bin/bash -e
SLURM_MAX_TASKS=1000  # Could be different- use `scontrol show config | grep MaxArraySize`

__cleanup() {
    scancel -u $USER
    skill -u $USER
}

send_out_batch() {
    array_end=$(($1 - 1)); shift
    command="sbatch --job-name=${2} --array=0-$array_end run_batch.sh $jobs_run $@"
    echo $command
    $command
}

trap __cleanup INT HUP TERM

job_desc=$1 ; shift
configs_file=$1 ; shift

lines_left=$(($(wc -l < $configs_file) - 1))
jobs_run=0
while [[ $lines_left -gt $SLURM_MAX_TASKS ]]; do
    send_out_batch $SLURM_MAX_TASKS $configs_file $job_desc
    jobs_run=$(( $jobs_run + $SLURM_MAX_TASKS ))
    lines_left=$(( $lines_left - $SLURM_MAX_TASKS ))
done

if [[ $lines_left -gt 0 ]]; then
    send_out_batch $lines_left $configs_file $job_desc
fi

echo make sure to remove $configs_file when you\'re done!
