import fnmatch
import numpy as np
import os
from collections import defaultdict

from constants import *
from utils import *


class TargetWord(object):
    """
    self.word: str, text of the target word
    self.index: int, the index of the target word, equivalent to targetWord2idx
    self.all_data: list(Data), contains the Data objects of this target word
    self.sense: dict(int, bool), contains the existing sense of this target word
    """
    idCounter = 0

    def __init__(self, word):
        self.word = word
        self.id = TargetWord.idCounter
        TargetWord.idCounter += 1
        self.label2target = {}
        self.target2label = {}

    def __str__(self):
        rst = "%s: " % self.word
        rst += "Id: %d; " % self.id
        rst += "Number of sense: %d" % len(self.label2target)
        return rst


class Data(object):
    """
    Data is a single data point, which includes the target word, target, and it's position
    """
    def __init__(self, target_word, n_doc, position, pos, target):
        """
        Data constructor
        self.target_word: TargetWord, refer to the target word of this data
        self.n_doc: int, index of the document
        self.position: int, index of the position in its document
        self.pos:
        self.label: int, index of the sense.
        """
        self.target_word = target_word
        self.target = target
        self.n_doc = n_doc
        self.position = position
        self.pos = pos

    def __str__(self):
        rst = "%s, in Doc %d, at %d, of meaning %d" % (
            self.target_word.word,
            self.n_doc,
            self.position,
            self.target
        )
        return rst


class DataSet(object):

    def __init__(self, dataset):
        """
        DataSet constructor
        self.dataset: list(Data)
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def add_data(self, data):
        """

        :param data: Data
        """
        self.dataset.append(data)

    def next(self, batch_size=32, shuffle=False):
        """
        Retrieve the next batch, use as generator
        :param batch_size: int
        :param shuffle: bool,
        :param restart: bool, restart the batch generation if reach bottom
        :return:
        """
        order = np.arange(len(self.dataset))
        cur = 0
        if shuffle:
            np.random.shuffle(order)
        while cur < len(self.dataset):
            next_batch = order[cur: cur + batch_size]
            yield [self.dataset[i] for i in next_batch]
            cur += batch_size


class BatchConstructor(object):
    """
    self.target_words: dict(str, TargetWord),
    self.datasets: dict(str, Dataset)

    Example: ["day", "show"]
        1. mixed:
            self.target_words: {"day": TargetWord(day); "show": TargetWord(show)}
            self.datasets: {"": Dataset(mixed)}
        2. not mixed:
            self.target_words: {"day": TargetWord(day); "show": TargetWord(show)}
            self.datasets: {"day": Dataset(day); "show": Dataset(show)}

    """
    def __init__(self, datasets, docList_indexed, context_size):
        self.datasets = datasets
        self.docList_indexed = docList_indexed
        self.context_size = context_size

    def __str__(self):
        rst = "Number of target documents: %d\n" % len(self.docList_indexed)
        rst += "Context size: %d" % self.context_size
        return rst

    def next(self, batch_size=32, shuffle=False):
        """
        Get the next batch of shuffled data, use as generator
        :param batch_size: size of data in a batch
        :param target_word:
        :param shuffle: shuffle the data or not
        :return:
        """
        left_buff = np.ndarray((batch_size, self.context_size), dtype=np.int32)
        right_buff = np.ndarray((batch_size, self.context_size), dtype=np.int32)
        dataset = self.datasets[mixed_str]
        for next_batch_data in dataset.next(batch_size, shuffle=shuffle):
            n_next_batch = len(next_batch_data)
            batch_target_words = [d.target_word for d in next_batch_data]
            batch_targets = [d.target for d in next_batch_data]

            for i, data in enumerate(next_batch_data):
                doc = self.docList_indexed[data.n_doc]
                position = data.position
                left_buff[i] = doc[position: position + self.context_size]
                right_buff[i] = doc[position + self.context_size + 1:position + 2 * self.context_size + 1]

            yield left_buff[:n_next_batch], right_buff[:n_next_batch], batch_target_words, batch_targets

    def next_by_target(self, batch_size, target_word, target, shuffle=False):
        """
        Return batch of data by word and its sense
        :param batch_size:
        :param target_word:
        :param target:
        :param shuffle:
        :return:
        """
        if target_word not in self.datasets:
            raise StopIteration

        dataset = self.datasets[target_word]
        left_buff = np.ndarray((batch_size, self.context_size), dtype=np.int32)
        right_buff = np.ndarray((batch_size, self.context_size), dtype=np.int32)
        data_of_target = np.array([data for data in dataset.dataset if data.target==target])

        if len(data_of_target) == 0:
            raise StopIteration

        order = np.arange(len(data_of_target))
        if shuffle:
            np.random.shuffle(order)
        cur = 0
        while cur < len(data_of_target):
            next_batch_data = data_of_target[order[cur: cur + batch_size]]
            n_next_batch = len(next_batch_data)
            for i, data in enumerate(next_batch_data):
                doc = self.docList_indexed[data.n_doc]
                position = data.position
                left_buff[i] = doc[position: position + self.context_size]
                right_buff[i] = doc[position + self.context_size + 1: position + 2 * self.context_size + 1]
            yield left_buff[:n_next_batch], right_buff[:n_next_batch]
            cur += batch_size
