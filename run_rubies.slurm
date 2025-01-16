#!/bin/bash
#SBATCH --ntasks 14 # The number of cores you need...
#SBATCH --job-name=rubies
#SBATCH --array=38 # Creates a job array for regions 01 to 39
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophie.newman@port.ac.uk
#SBATCH --output=./logs/log%A_%a.out
#SBATCH --error=./logs/log%A_%a.err

# Load modules
module load gnu_comp/14.1.0
module load openmpi/5.0.3
source ../dummy/bin/activate

# Set the directory
cd /cosma7/data/dp276/dc-newm1/flares_rubies_comp

echo $PWD

# Determine the region ID from the job array index
# and ensure the region is zero-padded to two digits, e.g. 02


REGION=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

# Run the pipeline
echo "Running for --region $REGION"

mpirun -np $SLURM_NTASKS python synthesize_rubies.py --grid bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps --grid-dir /cosma7/data/dp276/dc-newm1/data/synthesizer_data/grids --region $REGION --snap 3 --nthreads $SLURM_CPUS_PER_TASK /cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5
