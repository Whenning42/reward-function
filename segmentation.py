# Needs to support caching the segmentations and accessing them
# with a UID.

from collections import deque

# This class uses (N, C, H, W) image layout
class SkyrogueSegmenter:
    def __init__(self, cache = None):
        if cache is not None:
            self.cache = cache

    def Threshold(self, images):
        # Input images should be in RGB 0-1 format.
        assert images.size(1) == 3
        assert images.max() <= 1
        assert images.min() >= 0

        return (images[:, 0:1, :, :] == 1).double()

    def Segment(self, images):
        assert images.size(1) == 1
        assert images.max() <= 1
        assert images.min() >= 0

        queue = deque()
        y_pos = images.size(2) // 2
        for x in range(0, images.size(3)):
            queue.append()
