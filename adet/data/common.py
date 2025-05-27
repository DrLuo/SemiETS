# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
import numpy as np
import pickle
import random

import torch.utils.data
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from detectron2.utils.serialize import PicklableWrapper

from detectron2.data.common import MapDataset, AspectRatioGroupedDataset

__all__ = ["MMSemiSupDataset", "SemiSupDataset", "AspectRatioGroupedSemiSupDataset"]


class MMSemiSupDataset(data.IterableDataset):
    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: a tuple containing two iterable generators. （labeled and unlabeled data)
               Each element must be a dict with keys "width" and "height", which will be used
               to batch data.
            batch_size (list):
        """

        self.label_dataset = iter(dataset[0])
        self.unlabel_dataset = iter(dataset[1])
        self.batch_size_label = batch_size[0]
        self.batch_size_unlabel = batch_size[1]
        # print(self.batch_size_label, self.batch_size_unlabel)

    def __iter__(self):
        result = {'sup': [], 'unsup_teacher': [], 'unsup_student': []}
        sup_list, teacher_list, student_list = [], [], []
        while True:
            if len(sup_list) < self.batch_size_label:
                sup_list.append(next(self.label_dataset))
            else:
                result['sup'] = sup_list
            if len(teacher_list) < self.batch_size_unlabel:
                unsup_sample = next(self.unlabel_dataset)
                teacher_list.append(unsup_sample['weak'])
                student_list.append(unsup_sample['strong'])
            else:
                result['unsup_teacher'] = teacher_list
                result['unsup_student'] = student_list
            if len(result['sup']) == self.batch_size_label and \
                    len(result['unsup_teacher']) == self.batch_size_unlabel and \
                    len(result['unsup_teacher']) == self.batch_size_unlabel:
                yield result
                sup_list, teacher_list, student_list = [], [], []
                result['sup'], result['unsup_teacher'], result['unsup_student'] = \
                    [], [], []



        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

class MultiSourceDataset(data.IterableDataset):
    def __init__(self, dataset, batch_size, semi_format=False):
        # self.label_dataset, self.unlabel_dataset = dataset
        self.semi_format = semi_format
        self.semi_dataset = [iter(d) for d in dataset]
        self.semi_batch_size = batch_size
        assert len(dataset) == len(batch_size)
        self.num_source = len(dataset)
        self._buckets_per_source = [[] for _ in range(self.num_source)]
        self.fill_flag = [False for _ in range(self.num_source)]

    def reorganize_semi_inputs(self, batch_buffer):
        result = {'sup': [], 'unsup_teacher': [], 'unsup_student': []}
        for data in range(batch_buffer):
            if data['semi'] == 'sup':
                result['sup'].append(data)
            elif data['semi'] == 'unsup':
                result['unsup_teacher'].append(data['weak'])
                result['unsup_student'].append(data['strong'])
        return result

    def __iter__(self):
        batch_buffer = []

        while True:
            for source in range(self.num_source):
                if len(self._buckets_per_source[source]) < self.semi_batch_size[source]:
                    self._buckets_per_source[source].append(next(self.semi_dataset[source]))
                    if len(self._buckets_per_source[source]) == self.semi_batch_size[source]:
                        self.fill_flag[source] = True

                if all(self.fill_flag):
                    for s in range(self.num_source):
                        batch_buffer += self._buckets_per_source[s]
                        # self._buckets_per_source[source] = []
                        # self.fill_flag[source] = False
                    if self.semi_format:
                        batch_buffer = self.reorganize_semi_inputs(batch_buffer)
                    yield batch_buffer
                    batch_buffer = []
                    self._buckets_per_source = [[] for _ in range(self.num_source)]
                    self.fill_flag = [False for _ in range(self.num_source)]



class SemiSupDataset(data.IterableDataset):
    def __init__(self, dataset, batch_size):
        # self.label_dataset, self.unlabel_dataset = dataset
        self.semi_dataset = dataset
        self.semi_batch_size = batch_size
        assert len(dataset) == len(batch_size)
        # self.batch_size_label = batch_size[0]
        # self.batch_size_unlabel = batch_size[1]

    def __iter__(self):
        # Semi-supervised learning based on Teacher-student architecture requires three inputs
        result = {'sup': [], 'unsup_teacher': [], 'unsup_student': []}
        for source in range(len(self.semi_dataset)):
            if len(result['sup']) < self.semi_batch_size[source]:
                result['sup'].append(next(self.semi_dataset[source]))
            else:
                yield result
                result['sup'] = []
            if len(result['unsup_teacher']) < self.semi_batch_size[source]:
                unsup_sample = next(self.semi_dataset[source])
                result['unsup_teacher'].append(unsup_sample['weak'])
                result['unsup_student'].append(unsup_sample['strong'])
            else:
                yield result
                result['unsup_teacher'] = []
                result['unsup_student'] = []



class AspectRatioGroupedSemiSupDataset(AspectRatioGroupedDataset):

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: a tuple containing two iterable generators. （labeled and unlabeled data)
               Each element must be a dict with keys "width" and "height", which will be used
               to batch data.
            batch_size (list):
        """
        # super(AspectRatioGroupedSemiSupDataset, self).__init__()

        self.label_dataset, self.unlabel_dataset = dataset
        self.batch_size_label = batch_size[0]
        self.batch_size_unlabel = batch_size[1]

        self._label_buckets = [[] for _ in range(2)]
        self._label_buckets_key = [[] for _ in range(2)]
        self._unlabel_buckets = [[] for _ in range(2)]
        self._unlabel_buckets_key = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        label_bucket, unlabel_bucket = [], []
        for d_label, d_unlabel in zip(self.label_dataset, self.unlabel_dataset):
            # d_unlabel is a tuple with len = 2
            # d_unlabel[0] is with strong augmentation, d_unlabel[1] is with weak augmentation

            # because we are grouping images with their aspect ratio
            # label and unlabel buckets might not have the same number of data
            # i.e., one could reach batch_size, while the other is still not
            if len(label_bucket) != self.batch_size_label:
                w, h = d_label["width"], d_label["height"]
                label_bucket_id = 0 if w > h else 1
                label_bucket = self._label_buckets[label_bucket_id]
                label_bucket.append(d_label[0])
                label_buckets_key = self._label_buckets_key[label_bucket_id]
                label_buckets_key.append(d_label[1])

            if len(unlabel_bucket) != self.batch_size_unlabel:
                w, h = d_unlabel[0]["width"], d_unlabel[0]["height"]
                unlabel_bucket_id = 0 if w > h else 1
                unlabel_bucket = self._unlabel_buckets[unlabel_bucket_id]
                unlabel_bucket.append(d_unlabel[0])
                unlabel_buckets_key = self._unlabel_buckets_key[unlabel_bucket_id]
                unlabel_buckets_key.append(d_unlabel[1])

            # yield the batch of data until all buckets are full
            if (
                len(label_bucket) == self.batch_size_label
                and len(unlabel_bucket) == self.batch_size_unlabel
            ):
                # label_strong, label_weak, unlabed_strong, unlabled_weak
                yield (
                    label_bucket[:],
                    unlabel_bucket[:],
                )
                del label_bucket[:]
                del unlabel_bucket[:]






class AspectRatioGroupedMultiSourceDataset(data.IterableDataset):

    def __init__(self, dataset, batch_size, semi_format=False):
        """
        Args:
            dataset: a tuple containing two iterable generators. （labeled and unlabeled data)
               Each element must be a dict with keys "width" and "height", which will be used
               to batch data.
            batch_size (list):
        """
        # super(AspectRatioGroupedSemiSupDataset, self).__init__()
        self.semi_format = semi_format

        self.semi_dataset = [iter(d) for d in dataset]
        self.semi_batch_size = batch_size
        assert len(dataset) == len(batch_size)
        self.num_source = len(dataset)

        # organize by group
        self._buckets_per_group = [ [[] for _ in range(self.num_source)] for _ in range(2)]
        # self._sizes_per_group = [[0 for _ in range(self.num_source)] for _ in range(2)]
        self.fill_flag_per_group = [[False for _ in range(self.num_source)] for _ in range(2)]

        # organize by group
        self._buckets_per_source = [ [[] for _ in range(2)] for _ in range(self.num_source)]
        self._sizes_per_source = [[0 for _ in range(2)] for _ in range(self.num_source)]
        self.fill_flag_per_source = [[False for _ in range(2)] for _ in range(self.num_source)]

        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def reorganize_semi_inputs(self, batch_buffer):
        result = {'sup': [], 'unsup_teacher': [], 'unsup_student': []}
        for data in range(batch_buffer):
            if data['semi'] == 'sup':
                result['sup'].append(data)
            elif data['semi'] == 'unsup':
                result['unsup_teacher'].append(data['weak'])
                result['unsup_student'].append(data['strong'])
        return result

    def __iter__(self):

        batch_buffer = []
        while True:
            # select a source and get one data from source an
            for source in range(self.num_source):

                if all(self.fill_flag_per_source[source]) == False:
                    d = next(self.semi_dataset[source])
                    if d['semi'] == 'sup':
                        w, h = d["width"], d["height"]
                    elif d['semi'] == 'unsup':
                        w, h = d["weak"]["width"], d["weak"]["height"]
                    else:
                        raise NotImplementedError('semi_anno must be sup or unsup')
                    group_id = 0 if w > h else 1    # link to the group

                    # update the sample
                    bucket = self._buckets_per_group[group_id][source]
                    bucket.append(d)

                    self._sizes_per_source[source][group_id] += 1
                    # self._sizes_per_group[group_id][source] += 1

                    # update the flag, as a sample only belongs to one group and one source, just update corresponding flag
                    self.fill_flag_per_source[source] = \
                        [si >= self.semi_batch_size[source] for si in self._sizes_per_source[source]]
                    self.fill_flag_per_group[group_id] = \
                        [self._sizes_per_source[s][group_id] >= self.semi_batch_size[s] for s in range(self.num_source)]

                    # if samples in one group are full, form the batch from sources
                    if all(self.fill_flag_per_group[group_id]):
                        for s in range(self.num_source):
                            batch_buffer += self._buckets_per_group[group_id][s][:self.semi_batch_size[s]]
                            self._buckets_per_group[group_id][s] = self._buckets_per_group[group_id][s][self.semi_batch_size[s]:]
                            self._sizes_per_source[s][group_id] -= self.semi_batch_size[s]

                        # reset the flag for all the groups and sources
                        for s in range(self.num_source):
                            self.fill_flag_per_source[s] = \
                                [si >= self.semi_batch_size[s] for si in self._sizes_per_source[s]]
                        for g in range(2):
                            self.fill_flag_per_group[g] = \
                                [self._sizes_per_source[s][g] >= self.semi_batch_size[s] for s in
                                 range(self.num_source)]

                        if self.semi_format:
                            batch_buffer = self.reorganize_semi_inputs(batch_buffer)
                        yield batch_buffer
                        batch_buffer = []


