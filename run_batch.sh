#!/bin/bash
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=6:00:00 #Request runtime of 12 hours max
#SBATCH --ntasks-per-node=1 #Each node does one task
#SBATCH --cpus-per-task=1 #Each node gets one CPU
#SBATCH --output logs/output_%j.txt #redirect output to output_JOBID.txt
#SBATCH --error logs/error_%j.txt #redirect errors to error_JOBID.txt
#SBATCH --mem=16gb # Memory per processor
#SBATCH --partition=short #Use the short partition

if [[ ( ! -v SLURM_ARRAY_TASK_ID ) || ( -z $2 ) ]]; then
    echo "Usage: sbatch --array=0-<N> run_batch.sh <configs file> <batch name>"
    echo "(running of this script is typically handled by dispatch.sh)"
    exit 1
fi

module load python
source venv313/bin/activate
index=$(( $SLURM_ARRAY_TASK_ID + $1 ))
shift
python -m mergers_core.models.simulation_sweeps $index "$@"
