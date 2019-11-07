import pytest
import torch
from skyrogue_classifier import Threshold

def test_Threshold():
    # Pytorch uses N, C, H, W
    images = torch.zeros(2, 3, 2, 2)

    # First image
    # Red channel of first image
    images[0, 0, 0, 0] = 1
    images[0, 0, 1, 0] = 1
    # Green channel of first image
    images[0, 1, 0, 1] = 1

    # Second image
    # Red channel of second image
    images[1, 0, 1, 1] = 1
    # Blue channel of second image
    images[1, 1, 0, 0] = 1

    images = Threshold(images)

    # Threshold should return greyscale images
    assert images.size(1) == 1

    # First image asserts
    assert images[0, 0, 0, 0] == 1
    assert images[0, 0, 1, 0] == 1
    assert images[0, 0, 0, 1] == 0
    assert images[0, 0, 1, 1] == 0

    # Second image asserts
    assert images[1, 0, 1, 1] == 1
    assert images[1, 0, 0, 0] == 0
    assert images[1, 0, 0, 1] == 0
    assert images[1, 0, 1, 0] == 0

