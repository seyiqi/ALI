#!/bin/sh

module purge
module load python/intel/2.7.12 
module load theano/0.9.0
module load h5py/intel/2.7.0rc2
module load pillow/intel/4.0.0

SIZE=$1
LIM=$2
CLSL=$3

rm -rf "food-101-$SIZEx$SIZE"
mkdir "food-101-$SIZEx$SIZE"

PP="$PYTHONPATH:./fuel/:./picklable-itertools/"

PYTHONPATH=$PP python -u script.py $SIZE $LIM $CLSL
