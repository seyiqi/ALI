#!/bin/bash
#
#SBATCH --job-name=food64
#SBATCH --output=food64.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:30:00
#SBATCH --mem=32GB

sh do.sh 64 2000 2000

