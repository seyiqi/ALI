#!/bin/bash
#
#SBATCH --job-name=food32
#SBATCH --output=food32.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:30:00
#SBATCH --mem=32GB

sh do.sh 32 2000 2000

