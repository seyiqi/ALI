#!/bin/bash
#
#SBATCH --job-name=flowers_interpolate
#SBATCH --output=flowers_interpolate.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH -p gpu

module purge
module load cuda
module load cudnn
module load python/intel/2.7.12 
module load theano/0.9.0
module load h5py/intel/2.7.0rc2
module load pillow/intel/4.0.0
module load scikit-learn/intel/0.18.1

nvidia-smi

FUEL_CONFIG=fuelrc PYTHONPATH=$PYTHONPATH:. THEANORC=theanorc python \
  scripts/interpolate --save-path flowers_interpolate.png \
  ali_flowers_32x32_13.tar
