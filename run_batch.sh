#!/bin/bash
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=12:00:00 #Request runtime of 12 hours max
#SBATCH --ntasks-per-node=1 #Each node does one task
#SBATCH --cpus-per-task=1 #Each node gets one CPU
#SBATCH --output logs/output_%j.txt #redirect output to output_JOBID.txt
#SBATCH --error logs/error_%j.txt #redirect errors to error_JOBID.txt
#SBATCH --mem=16gb # Memory per processor
#SBATCH --partition=short #Use the short partition
#SBATCH --mail-type=BEGIN,END #Mail when job starts and ends
#SBATCH --mail-user=se.gracia@northeastern.edu #email recipient
#SBATCH --array=0-499 #Run all jobs in parallel

# Make sure you pass the batch name to run.
module load python
source venv313/bin/activate
python -m mergers_core.models.simulation_sweeps $SLURM_ARRAY_TASK_ID 500 $1
