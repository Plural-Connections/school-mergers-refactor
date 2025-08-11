#!/bin/bash -e
SLURM_MAX_TASKS=1000  # Could be different- use `scontrol show config | grep MaxArraySize`

__cleanup() {
    scancel -u $USER
    skill -u $USER
    rm $full_file
}

send_out_batch() {
    array_end=$1; shift
    dependency_id=$1; shift
    if [[ $dependency_id = "NONE" ]]; then
        dependency=""
    else
        dependency="--dependency=afterany:$dependency_id"
    fi
    command="sbatch --job-name=${2} --array=0-$array_end run_batch.sh $jobs_run $@"
    echo $command
    sbatch_output=$($command)
    echo $sbatch_output
    # sbatch's output looks like: 'Submitted batch job 12345'
    read _ _ _ job_id <<< $sbatch_output
    echo $job_id >&2
}

trap __cleanup INT HUP TERM

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

full_file=data/sweep_configs/merged_$batchname
rm -f $full_file && touch $full_file
cat "${batch_dirs[@]}" > $full_file

exec 3>&1  # fd 3 redirects to stdout

lines_left=$(wc -l < $full_file)
jobs_run=0
all_deps=''
dependency="NONE"
while [[ $lines_left -gt $SLURM_MAX_TASKS ]]; do
    dependency=$(send_out_batch $SLURM_MAX_TASKS "$dependency" $full_file $batchname 2>&1 1>&3)
    all_deps+="afterany:$dependency:"
    jobs_run=$(( $jobs_run + $SLURM_MAX_TASKS ))
    lines_left=$(( $lines_left - $SLURM_MAX_TASKS ))
done

if [[ $lines_left -gt 0 ]]; then
    dependency=$(send_out_batch $lines_left $dependency $full_file $batchname 2>&1 1>&3)
    all_deps+="afterany:$dependency:"
fi

exec 3>&-

wait_dummy="srun --dependency=${all_deps::-1} echo done"
echo $wait_dummy $action
# $wait_dummy "$action"
# rm $full_file
