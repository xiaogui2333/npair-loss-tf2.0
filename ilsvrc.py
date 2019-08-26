# -*- coding: utf-8 -*-

import numpy as np
import os

from tensorpack.dataflow.base import RNGDataFlow


class ILSVRC12Files(RNGDataFlow):
    """
    Same as :class:`ILSVRC12`, but produces filenames of the images instead of nparrays.
    This could be useful when ``cv2.imread`` is a bottleneck and you want to
    decode it in smarter ways (e.g. in parallel).
    """
    def __init__(self, dir, name, meta_dir=None,
                 shuffle=None, dir_structure=None):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.shuffle = shuffle
        self.imglist = dir

    def __len__(self):
        return len(self.imglist)

    def __iter__(self):
        idxs = np.arange(len(self.imglist))
        # if self.shuffle:
        #     self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.imglist[k]
            # fname = os.path.join(self.full_dir, fname)
            yield [fname, label]