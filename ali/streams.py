"""Functions for creating data streams."""
from fuel.datasets import CIFAR10, SVHN, CelebA
from fuel.datasets.toy import Spiral
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.datasets import H5PYDataset
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.utils import find_in_data_path

from .datasets import TinyILSVRC2012, GaussianMixture


def create_svhn_data_streams(batch_size, monitoring_batch_size, rng=None):
    train_set = SVHN(2, ('extra',), sources=('features',))
    valid_set = SVHN(2, ('train',), sources=('features',))
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        valid_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_cifar10_data_streams(batch_size, monitoring_batch_size, rng=None):
    train_set = CIFAR10(
        ('train',), sources=('features',), subset=slice(0, 45000))
    valid_set = CIFAR10(
        ('train',), sources=('features',), subset=slice(45000, 50000))
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        valid_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


class Flower(H5PYDataset):
    filename = 'flowers102_32x32.hdf5'
    default_transformers = uint8_pixels_to_floatX(('features',))

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(Flower, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)


def create_flower_data_streams(
        batch_size, monitoring_batch_size, rng=None,
        train_slice=slice(0, 114646),
        valid_slice=slice(114646, 122835)):
    # Since it's so small just use the entire dataset.
    train_set = Flower(
        ('train',), sources=('features',), subset=slice(0, 114646))
    valid_set = Flower(
        ('train',), sources=('features',), subset=slice(114646, 122835))
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        valid_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_celeba_data_streams(batch_size, monitoring_batch_size,
                               sources=('features', ), rng=None):
    train_set = CelebA('64', ('train',), sources=sources)
    valid_set = CelebA('64', ('valid',), sources=sources)
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        valid_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_tiny_imagenet_data_streams(batch_size, monitoring_batch_size,
                                      rng=None):
    train_set = TinyILSVRC2012(('train',), sources=('features',))
    valid_set = TinyILSVRC2012(('valid',), sources=('features',))
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            4096, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        valid_set,
        iteration_scheme=ShuffledScheme(
            4096, monitoring_batch_size, rng=rng))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_spiral_data_streams(batch_size, monitoring_batch_size, rng=None,
                               num_examples=100000, classes=1, cycles=2,
                               noise=0.1):
    train_set = Spiral(num_examples=num_examples, classes=classes,
                       cycles=cycles, noise=noise, sources=('features',))

    valid_set = Spiral(num_examples=num_examples, classes=classes,
                       cycles=cycles, noise=noise, sources=('features',))

    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size=batch_size, rng=rng))

    train_monitor_stream = DataStream.default_stream(
        train_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    valid_monitor_stream = DataStream.default_stream(
        valid_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_gaussian_mixture_data_streams(batch_size, monitoring_batch_size,
                                         means=None, variances=None, priors=None,
                                         rng=None, num_examples=100000,
                                         sources=('features', )):
    train_set = GaussianMixture(num_examples=num_examples, means=means,
                                variances=variances, priors=priors,
                                rng=rng, sources=sources)

    valid_set = GaussianMixture(num_examples=num_examples,
                                means=means, variances=variances,
                                priors=priors, rng=rng, sources=sources)

    main_loop_stream = DataStream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size=batch_size, rng=rng))

    train_monitor_stream = DataStream(
        train_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    valid_monitor_stream = DataStream(
        valid_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    return main_loop_stream, train_monitor_stream, valid_monitor_stream
