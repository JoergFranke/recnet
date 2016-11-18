#!/usr/bin/env python
"""
This script generates a data set with images of numbers for the numbers recognition task.

Credits: I used and modified software from Rakesh Var under Apache 2 License.
https://www.apache.org/licenses/LICENSE-2.0

"""

import os
import klepto
import numpy as np

numbers_ = [
    [
        [0,1,1,0],
        [1,0,0,1],
        [1,0,0,1],
        [0,1,1,0],
    ],
    [
        [1,1,0],
        [0,1,0],
        [0,1,0],
        [1,1,1],
    ],
    [
        [0,1,1,0],
        [1,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
        [1,1,1,1],
    ],
    [
        [0,1,1,0],
        [1,0,0,1],
        [0,0,1,0],
        [1,0,0,1],
        [0,1,1,0],
    ],
    [
        [0,0,1,0,0,1],
        [0,1,0,0,1,0],
        [1,1,1,1,0,0],
        [0,0,1,0,0,0],
        [0,1,0,0,0,0],
    ],
    [
        [1,1,1],
        [1,0,0],
        [1,1,1],
        [0,0,1],
        [1,0,1],
        [1,1,1],
    ],
    [
        [0,1,1,0],
        [1,0,0,0],
        [1,1,1,0],
        [1,0,0,1],
        [0,1,1,0],
    ],
    [
        [1,1,1,1,1],
        [0,0,0,1,0],
        [0,0,1,0,0],
        [0,1,1,1,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
    ],
    [
        [0,1,1,0],
        [1,0,0,1],
        [0,1,1,0],
        [1,0,0,1],
        [0,1,1,0],
    ],
    [
        [0,1,1,0],
        [1,0,0,1],
        [0,1,1,1],
        [0,0,0,1],
        [0,0,1,0],
    ]

]

numbers = [np.asarray(num) for num in numbers_]


class Alphabet():
    def __init__(self, chars=None):
        self.bitmaps = [np.asarray(num) for num in numbers_] #bitmaps
        self.n = len(self.bitmaps)
        self.chars = chars if chars else [str(i) for i in range(len(self.bitmaps))]

        self.maxHt = max([bitmap.shape[0] for bitmap in self.bitmaps])
        self.maxWd = max([bitmap.shape[1] for bitmap in self.bitmaps])

    def get_char(self, index=None):
        if index is None:
            index = np.random.choice(self.n)

        bitmap = self.bitmaps[index]
        char = self.chars[index]
        return index, char, bitmap

    def __str__(self):
        ret = ''
        for c, b in zip(self.chars, self.bitmaps):
            slab = '\n'.join((''.join('# '[p] for p in r) for r in b))
            ret += '\n{}:\n{}'.format(c, slab)
        return ret


class Scribe():
    def __init__(self, parameter):
        self.alphabet = Alphabet()    #parameter["alphabet"]
        self.noise = parameter["noise"]
        self.hbuffer = parameter["hbuffer"]
        self.vbuffer = parameter["vbuffer"]

        self.avg_len = parameter["avg_seq_len"]
        self.varying_len = parameter["varying_len"]
        self.nchars_per_sample = parameter["nchars_per_sample"]

        self.nDims = self.alphabet.maxHt + self.vbuffer

        self.shuffled_char_indices = np.arange(self.alphabet.n)
        self.shuffled_char_pointer = 0

        self.nClasses = len(self.alphabet.chars)
        self.len_range = (3*self.avg_len//4,  5*self.avg_len//4)

    def get_sample_of_random_len(self):
        length = np.random.randint(*self.len_range)
        return self.get_sample_of_len(length)

    def get_next_char(self):
        if self.shuffled_char_pointer == len(self.shuffled_char_indices):
            np.random.shuffle(self.shuffled_char_indices)
            self.shuffled_char_pointer = 0

        self.shuffled_char_pointer += 1
        return self.alphabet.get_char(self.shuffled_char_indices[
            self.shuffled_char_pointer-1])

    def get_sample_of_len(self, length):
        image = np.zeros((self.nDims, length), dtype=float)
        labels = []

        at_wd = np.random.exponential(self.hbuffer) + self.hbuffer
        while at_wd < length - self.hbuffer - self.alphabet.maxWd:
            index, char, bitmap = self.get_next_char()
            ht, wd = bitmap.shape
            at_ht = np.random.randint(self.vbuffer + self.alphabet.maxHt - ht + 1)
            image[at_ht:at_ht+ht, at_wd:at_wd+wd] += bitmap
            at_wd += wd + np.random.randint(self.hbuffer)
            labels.append(index)

        image += self.noise * np.random.normal(size=image.shape,)
        image = np.clip(image, 0, 1)
        return image, labels

    def get_sample_of_n_chars(self, n):
        gaps = np.random.randint(self.hbuffer, size=n+1)
        labels_bitmaps = [self.get_next_char() for _ in range(n)]
        labels, _, bitmaps = zip(*labels_bitmaps)
        length = sum(gaps) + sum(b.shape[1] for b in bitmaps)
        image = np.zeros((self.nDims, length), dtype=float)

        at_wd = gaps[-1]
        for i, bitmap in enumerate(bitmaps):
            ht, wd = bitmap.shape
            at_ht = np.random.randint(self.vbuffer + self.alphabet.maxHt - ht + 1)
            image[at_ht:at_ht+ht, at_wd:at_wd+wd] += bitmap
            at_wd += wd + gaps[i]

        image += self.noise * np.random.normal(size=image.shape,)
        image = np.clip(image, 0, 1)
        return image, labels

    def get_sample(self):
        if self.nchars_per_sample:
            return self.get_sample_of_n_chars(self.nchars_per_sample)

        if self.varying_len:
            return self.get_sample_of_random_len()
        else:
            return self.get_sample_of_len(self.avg_len)

    def __repr__(self):
        if self.nchars_per_sample:
            cps = self.nchars_per_sample
            len = 'Varies (to fit the {} of chars per sample)'.format(cps)
        else:
            cps = 'Depends on the random length'
            len = 'Avg:{} Range:{}'.format(self.avg_len, self.len_range)

        ret = ('Scribe:'
               '\n  Alphabet: {}'
               '\n  Noise: {}'
               '\n  Buffers (vert, horz): {}, {}'
               '\n  Characters per sample: {}'
               '\n  Length: {}'
               '\n  Height: {}'
               '\n'.format(
            ''.join(self.alphabet.chars),
            self.noise,
            self.hbuffer,
            self.vbuffer,
            cps, len,
            self.nDims,
            ))
        return ret


######                              Main
########################################
if __name__ == "__main__":

    # create data set folder
    if "data_set" not in os.listdir(os.getcwd()):
        os.mkdir("data_set")

    out_file_name = "data_name"
    if not out_file_name.endswith('.pkl'):
        out_file_name += '.pkl'

    scribe_args = {
        'noise': .05,
        'vbuffer': 3,
        'hbuffer': 5,
        'avg_seq_len': 5,
        'varying_len': True,
        'nchars_per_sample': 10,
        'num_samples':2000,
        }

    scriber = Scribe(scribe_args)
    alphabet_chars = scriber.alphabet.chars

    xs = []
    ys = []
    for i in range(scribe_args['num_samples']):
        x, y = scriber.get_sample()
        xs.append(np.transpose(x))
        ys.append(np.asarray(y))
        print(y, ''.join(alphabet_chars[i] for i in y))


    print("write train set")
    print("train set length: " + str(ys.__len__()))
    file_name = "data_set/numbers_image_train.klepto"
    print("train set name: " + file_name)
    d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
    d['x'] = xs
    d['y'] = ys
    d.dump()
    d.clear()


    xs = []
    ys = []
    for i in range(int(scribe_args['num_samples'] * 0.1)):
        x, y = scriber.get_sample()
        xs.append(np.transpose(x))
        ys.append(np.asarray(y))
        print(y, ''.join(alphabet_chars[i] for i in y))


    print("write valid set")
    print("valid set length: " + str(ys.__len__()))
    file_name = "data_set/numbers_image_valid.klepto"
    print("valid set name: " + file_name)
    d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
    d['x'] = xs
    d['y'] = ys
    d.dump()
    d.clear()

    xs = []
    ys = []
    for i in range(int(scribe_args['num_samples'] * 0.3)):
        x, y = scriber.get_sample()
        xs.append(np.transpose(x))
        ys.append(np.asarray(y))
        print(y, ''.join(alphabet_chars[i] for i in y))


    print("write test set")
    print("test set length: " + str(ys.__len__()))
    file_name = "data_set/numbers_image_test.klepto"
    print("test set name: " + file_name)
    d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
    d['x'] = xs
    d['y'] = ys
    d.dump()
    d.clear()




