#!/bin/bash
#
#SBATCH --job-name=flowers_sample
#SBATCH --output=flowers_sample.out
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
  scripts/gan_generate.py \
  --save-path flowers_sample.png \
  ali_flowers_32x32_13.tar
