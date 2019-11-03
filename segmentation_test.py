# Using set colorcolumn=65 because it seems nice.
import pytest
import torch
from segmentation import SkyrogueSegmenter

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

    segmenter = SkyrogueSegmenter(cache = False)
    images = segmenter.Threshold(images)

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

def test_Segment():
    images = torch.zeros(3, 1, 10, 10)

    # Squares in the middle of the image
    # vertically at either side of the image.
    images[0, 0, 4:7, 0:3] = 1
    images[0, 0, 4:7, 7:10] = 1

    # Bands on the top and bottom of the image
    # that go all the way across.
    images[1, 0, 0:3, 0:10] = 1
    images[1, 0, 7:10, 0:10] = 1

    # Check that we count adjacent diagonally as connected.
    # This image is one block in the middle vertically that
    # diagonally touches another block that is above the middle.
    images[1, 0, 4:7, 4:7] = 1
    images[1, 0, 0:4, 7:10] = 1
