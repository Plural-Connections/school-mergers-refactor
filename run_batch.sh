#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --ntasks-per-node=1  # Singleton jobs
#SBATCH --cpus-per-task=4  # Same as in constants.py
#SBATCH --time=6:00:00
#SBATCH --output logs/output_%j.txt --error logs/error_%j.txt
#SBATCH --mem=16gb
#SBATCH --partition=short  # Use the short partition

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
