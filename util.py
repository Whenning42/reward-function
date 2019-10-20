import matplotlib.pyplot as plt
import numpy as np
import time

# This function can be called repeatedly to keep updating the displayed image
# as opposed to PIL's Image.show() which breaks in that case.
def DisplayImage(image, blocking = True, block_for = None):
    assert(isinstance(image, (np.ndarray, np.generic))), \
            "DisplayImage was passed something that isn't an np array"
    image = image / 255
    plt.imshow(image)
    plt.draw()
    plt.pause(.001)
    if block_for:
        time.sleep(block_for)
    elif blocking:
        input()

# pyplots hist behavior is weird when trying to plot prebinned data. To get
# expected behavior for prebinned data, you need to pass in one too many bins
# and have the last element of the passed in binned data be zero.
def ShowPrebinnedHistogram(prebinned_data, title = None):
    data_with_extra_bin = np.zeros(prebinned_data.shape[0] + 1)
    data_with_extra_bin[:-1] = prebinned_data
    bins = data_with_extra_bin.shape[0]
    plt.hist(range(0, bins), range(0, bins), weights = data_with_extra_bin)
    if title is not None:
        plt.title(title)
    plt.show()
