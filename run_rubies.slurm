#!/bin/bash
#SBATCH --ntasks 28 # The number of cores you need...
#SBATCH --job-name=rubies
#SBATCH -p cosma7-prince
#SBATCH -A dp276
#SBATCH --cpus-per-task=1
#SBATCH -t 05:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophie.newman@port.ac.uk
#SBATCH --output=./logs/log%A_%a.out
#SBATCH --error=./logs/log%A_%a.err

# Set the directory
cd /cosma7/data/dp276/dc-newm1/flares_rubies_comp

echo $PWD

# Run the pipeline
mpirun -np $SLURM_NTASKS python synthesize_rubies.py --grid bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps --grid-dir /cosma7/data/dp276/dc-newm1/data/synthesizer_data/grids --region 00 --snap 3 --nthreads 28 /cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5
