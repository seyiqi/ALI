#!/bin/bash
#
#SBATCH --job-name=foodtest
#SBATCH --output=foodtest.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --mem=16GB

sh do.sh 64 10 17

