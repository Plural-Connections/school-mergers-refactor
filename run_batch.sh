#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --ntasks-per-node=1  # Singleton jobs
#SBATCH --cpus-per-task=4  # Same as in constants.py
#SBATCH --time=6:00:00
#SBATCH --output logs/output_%j.txt --error logs/error_%j.txt
#SBATCH --mem=16gb
#SBATCH --partition=short  # Use the short partition

if [[ ( ! -v SLURM_ARRAY_TASK_ID ) || ( -z $2 ) ]]; then
    echo "Usage: sbatch --array=0-<N> run_batch.sh <batch starting index> <filename>"
    echo "(running of this script is typically handled by dispatch.sh)"
    exit 1
fi

module load python
source venv313/bin/activate
index=$(( $SLURM_ARRAY_TASK_ID + $1 )) ; shift
filename=$1 ; shift
python <<EOF
import mergers_core.models.config as config
from mergers_core.models.merge_cp_sat import solve_and_output_results

print("task id: $SLURM_ARRAY_TASK_ID; index = $index; filename = $filename")
solve_and_output_results(config.Config("$filename", entry_index=$index))
EOF
