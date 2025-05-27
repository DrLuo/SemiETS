# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import math
import numpy as np

from collections import defaultdict
from typing import Iterator, List, Optional, Sized, Union
import torch
from torch.utils.data.sampler import Sampler

from detectron2.utils import comm

logger = logging.getLogger(__name__)


class MultiSourceSampler(Sampler):
    r"""Multi-Source Infinite Sampler.

    According to the sampling ratio, sample data from different
    datasets to form batches.

    Args:
        dataset (Sized): The dataset.
        batch_size (int): Size of mini-batch.
        source_ratio (list[int | float]): The sampling ratio of different
            source datasets in a mini-batch.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.

    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 source_ratio: List[Union[int, float]],
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:

        assert hasattr(dataset, 'cumulative_sizes'),\
            f'The dataset must be ConcatDataset, but get {dataset}'
        assert isinstance(batch_size, int) and batch_size > 0, \
            'batch_size must be a positive integer value, ' \
            f'but got batch_size={batch_size}'
        assert isinstance(source_ratio, list), \
            f'source_ratio must be a list, but got source_ratio={source_ratio}'
        assert len(source_ratio) == len(dataset.cumulative_sizes), \
            'The length of source_ratio must be equal to ' \
            f'the number of datasets, but got source_ratio={source_ratio}'


        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()

        self.dataset = dataset
        self.cumulative_sizes = [0] + dataset.cumulative_sizes
        self.batch_size = batch_size
        self.source_ratio = source_ratio

        self.num_per_source = [
            int(batch_size * sr / sum(source_ratio)) for sr in source_ratio
        ]
        self.num_per_source[0] = batch_size - sum(self.num_per_source[1:])

        assert sum(self.num_per_source) == batch_size, \
            'The sum of num_per_source must be equal to ' \
            f'batch_size, but get {self.num_per_source}'

        seed = comm.shared_random_seed() if seed is None else seed
        self.seed = int(seed)
        self.shuffle = shuffle
        self.source2inds = {
            source: self._indices_of_rank(len(ds))
            for source, ds in enumerate(dataset.datasets)
        }

    def _infinite_indices(self, sample_size: int) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(sample_size, generator=g).tolist()
            else:
                yield from torch.arange(sample_size).tolist()

    def _indices_of_rank(self, sample_size: int) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(
            self._infinite_indices(sample_size), self.rank, None,
            self.world_size)

    def __iter__(self) -> Iterator[int]:
        batch_buffer = []
        while True:
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.source2inds[source]:
                    idx += self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
            yield from batch_buffer
            batch_buffer = []

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """Not supported in `epoch-based runner."""
        pass


class GroupMultiSourceSampler(MultiSourceSampler):
    r"""Group Multi-Source Infinite Sampler.

    According to the sampling ratio, sample data from different
    datasets but the same group to form batches.

    Args:
        dataset (Sized): The dataset.
        batch_size (int): Size of mini-batch.
        source_ratio (list[int | float]): The sampling ratio of different
            source datasets in a mini-batch.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """

    def __init__(self,
                 dataset,
                 batch_size: int,
                 source_ratio: List[Union[int, float]],
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            source_ratio=source_ratio,
            shuffle=shuffle,
            seed=seed)

        self._get_source_group_info()
        self.group_source2inds = [{
            source:
            self._indices_of_rank(self.group2size_per_source[source][group])
            for source in range(len(dataset.datasets))
        } for group in range(len(self.group_ratio))]

    def _get_source_group_info(self) -> None:
        self.group2size_per_source = [{0: 0, 1: 0}, {0: 0, 1: 0}]
        self.group2inds_per_source = [{0: [], 1: []}, {0: [], 1: []}]
        for source, dataset in enumerate(self.dataset.datasets):
            for idx in range(len(dataset)):
                data_info = dataset.get_data_info(idx)
                width, height = data_info['width'], data_info['height']
                group = 0 if width < height else 1
                self.group2size_per_source[source][group] += 1
                self.group2inds_per_source[source][group].append(idx)

        self.group_sizes = np.zeros(2, dtype=np.int64)
        for group2size in self.group2size_per_source:
            for group, size in group2size.items():
                self.group_sizes[group] += size
        self.group_ratio = self.group_sizes / sum(self.group_sizes)

    def __iter__(self) -> Iterator[int]:
        batch_buffer = []
        while True:
            group = np.random.choice(
                list(range(len(self.group_ratio))), p=self.group_ratio)
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.group_source2inds[group][source]:
                    idx = self.group2inds_per_source[source][group][
                        idx] + self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
            yield from batch_buffer
            batch_buffer = []



class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)

    Note that this sampler does not shard based on pytorch DataLoader worker id.
    A sampler passed to pytorch DataLoader is used only with map-style dataset
    and will not be executed inside workers.
    But if this sampler is used in a way that it gets execute inside a dataloader
    worker, then extra work needs to be done to shard its outputs based on worker id.
    This is required so that workers don't produce identical data.
    :class:`ToIterableDataset` implements this logic.
    This note is true for all samplers in detectron2.
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        if not isinstance(size, int):
            raise TypeError(f"TrainingSampler(size=) expects an int. Got type {type(size)}.")
        if size <= 0:
            raise ValueError(f"TrainingSampler(size=) expects a positive int. Got {size}.")
        self._size = size
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                yield from torch.arange(self._size).tolist()


class RandomSubsetTrainingSampler(TrainingSampler):
    """
    Similar to TrainingSampler, but only sample a random subset of indices.
    This is useful when you want to estimate the accuracy vs data-number curves by
      training the model with different subset_ratio.
    """

    def __init__(
        self,
        size: int,
        subset_ratio: float,
        shuffle: bool = True,
        seed_shuffle: Optional[int] = None,
        seed_subset: Optional[int] = None,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            subset_ratio (float): the ratio of subset data to sample from the underlying dataset
            shuffle (bool): whether to shuffle the indices or not
            seed_shuffle (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
            seed_subset (int): the seed to randomize the subset to be sampled.
                Must be the same across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        super().__init__(size=size, shuffle=shuffle, seed=seed_shuffle)

        assert 0.0 < subset_ratio <= 1.0
        self._size_subset = int(size * subset_ratio)
        assert self._size_subset > 0
        if seed_subset is None:
            seed_subset = comm.shared_random_seed()
        self._seed_subset = int(seed_subset)

        # randomly generate the subset indexes to be sampled from
        g = torch.Generator()
        g.manual_seed(self._seed_subset)
        indexes_randperm = torch.randperm(self._size, generator=g)
        self._indexes_subset = indexes_randperm[: self._size_subset]

        logger.info("Using RandomSubsetTrainingSampler......")
        logger.info(f"Randomly sample {self._size_subset} data from the original {self._size} data")

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)  # self._seed equals seed_shuffle from __init__()
        while True:
            if self._shuffle:
                # generate a random permutation to shuffle self._indexes_subset
                randperm = torch.randperm(self._size_subset, generator=g)
                yield from self._indexes_subset[randperm].tolist()
            else:
                yield from self._indexes_subset.tolist()


class RepeatFactorTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but a sample may appear more times than others based
    on its "repeat factor". This is suitable for training on class imbalanced datasets like LVIS.
    """

    def __init__(self, repeat_factors, *, shuffle=True, seed=None):
        """
        Args:
            repeat_factors (Tensor): a float vector, the repeat factor for each indice. When it's
                full of ones, it is equivalent to ``TrainingSampler(len(repeat_factors), ...)``.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        # Split into whole number (_int_part) and fractional (_frac_part) parts.
        self._int_part = torch.trunc(repeat_factors)
        self._frac_part = repeat_factors - self._int_part

    @staticmethod
    def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors based on category frequency.
        The repeat factor for an image is a function of the frequency of the rarest
        category labeled in that image. The "frequency of category c" in [0, 1] is defined
        as the fraction of images in the training set (without repeats) in which category c
        appears.
        See :paper:`lvis` (>= v2) Appendix B.2.

        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
                If the frequency is half of `repeat_thresh`, the image will be
                repeated twice.

        Returns:
            torch.Tensor:
                the i-th element is the repeat factor for the dataset image at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids}, default=1.0)
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm].tolist()
            else:
                yield from indices.tolist()



