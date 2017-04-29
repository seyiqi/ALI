#!/usr/bin/python

import os
import numpy as np
import re
import h5py
import sys
import re
from PIL import Image, ImageOps
from fuel.converters.base import fill_hdf5_file
from scipy.misc import imread

size = int(sys.argv[1])
limit = int(sys.argv[2])
cls_limit = int(sys.argv[3])

labels = os.listdir('food-101/images')
label_ids = range(len(labels))

with open('food-101-labels-{0}x{0}.txt'.format(size), 'w') as f:
    for l in labels:
        f.write(l)
        f.write('\n')

def read_in(size=64, limit=10, class_limit=10):
    all_vs = None
    vectors = []
    targets = []
    for i, c in zip(label_ids, labels)[:class_limit]:
        print(c)
        for j, fname in enumerate(os.listdir('food-101/images/' + c)[:limit]):
            infile = os.path.join('food-101/images', c, fname)
            outfname = re.sub('jpe?g$', 'png', fname)
            outfile = 'food-101-{0}x{0}/{1}_{2}'.format(size, c, outfname)
            try:
                im = Image.open(infile)
                im = ImageOps.fit(
                    im,
                    (size, size),
                    Image.ANTIALIAS
                )
                if j < 10:
                    # Save the sample 10 images
                    im.save(outfile, "png")
                im = np.array(im, dtype=np.uint8)
                if len(im.shape) != 3:
                    continue
                imarr = im.transpose([2, 0, 1])
                if imarr.shape != (3, size, size):
                    continue
                targets.append(i)
                vectors.append(imarr)
                if len(vectors) >= 10000:
                    if all_vs is None:
                        all_vs = np.stack(vectors, 0)
                    else:
                        vs = np.stack(vectors, 0)
                        all_vs = np.concatenate([all_vs, vs], 0)
                    vectors = []
            except IOError:
                print "cannot create thumbnail for '%s'" % infile
                    
    vectors = np.stack(vectors, 0)
    if all_vs is not None:
        vectors = np.concatenate([all_vs, vectors], 0)
    targets = np.array(targets).reshape(-1, 1)
    assert len(vectors) == len(targets)
    
    data = (('train', 'features', vectors),
            ('train', 'targets', targets))
    
    h5file = h5py.File('food-101-{0}x{0}.hdf5'.format(size), mode='w')
    try:
        fill_hdf5_file(h5file, data)

        h5file['features'].dims[0].label = 'batch'
        h5file['features'].dims[1].label = 'channel'
        h5file['features'].dims[2].label = 'height'
        h5file['features'].dims[3].label = 'width'
        h5file['targets'].dims[0].label = 'batch'
        h5file['targets'].dims[1].label = 'index'

        h5file.flush()
    finally:
        h5file.close()
        
    return vectors, targets
    print('Done!')

read_in(size, limit, cls_limit)
