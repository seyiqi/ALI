To run this model, make sure the following programs are installed:
- theano
- blocks
- fuel
- gpuarray
- pygpu

Instructions to install pygpu can be found in
http://deeplearning.net/software/libgpuarray/installation.html#step-by-step-install-user-library

After installing the prerequisites, download the models and data:
  - download the 'ali_flowers_32x32_11_e206.tar' from Google Drive.
  - download the 'flowers102_32x32.hdf5' (dataset for interpolations) from Google Drive.
Place them on the ALI directory.

Flower model:
https://drive.google.com/open?id=0B3PT5wpPFOx7UUJYWGIxR250bUE
Flower dataset:
https://drive.google.com/open?id=0B3PT5wpPFOx7cWw0dXIxbDh0R1k

To generate interpolation plots, run
$ do_interpolate.sh

To generate new samples, run
$ do_sample.sh

You can also run scripts/gan_generate.py manually.
