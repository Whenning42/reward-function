import pytest
import torch
from segmentation import SkyrogueSegmenter, FilterOutlierSegments, CenterImages

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

    segmenter = SkyrogueSegmenter()
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

    # Squares in the middle of the image vertically at either side of the
    # image.
    images[0, 0, 4:7, 0:3] = 1
    images[0, 0, 4:7, 7:10] = 1

    # Bands on the top and bottom of the image that go all the way across.
    images[1, 0, 0:3, 0:10] = 1
    images[1, 0, 7:10, 0:10] = 1

    # One block that is diagonally adjacent to another. This first block sits
    # along the middle band of the image.
    images[2, 0, 4:7, 4:7] = 1
    images[2, 0, 1:4, 7:10] = 1

    segmenter = SkyrogueSegmenter()
    segments = segmenter.Segment(images)

    assert len(segments) == 3

    assert len(segments[0]) == 2
    assert segments[0][0].size() == (3, 3)
    assert segments[0][1].size() == (3, 3)

    assert len(segments[1]) == 0

    assert len(segments[2]) == 1
    assert segments[2][0].size() == (6, 6)

def test_Filter():
    # Images are (H, W) format
    segmentations = [[torch.zeros(4, 4), torch.zeros(4, 4)], \
                     [torch.zeros(20, 20), torch.zeros(4, 4)], \
                     [torch.zeros(4, 10)], \
                     [torch.zeros(10, 4)],
                     []]
    filter = FilterOutlierSegments(min_width = 2, max_width = 8, min_height = 2, max_height = 12)
    filter.FilterInPlace(segmentations)

    assert len(segmentations) == 5
    assert len(segmentations[0]) == 2
    assert len(segmentations[1]) == 0
    assert len(segmentations[2]) == 0
    assert len(segmentations[3]) == 1
    assert len(segmentations[4]) == 0

def test_Centering():
    kNewSize = (20, 10)
    segmentations = [[torch.ones(4, 8)]]

    centerer = CenterImages(kNewSize)
    centerer.ResizeInPlace(segmentations)
    assert segmentations[0][0].size() == kNewSize
    assert segmentations[0][0][10, 5] == 1
    assert segmentations[0][0][15, 5] == 0
    assert segmentations[0][0][10, 7] == 1
