# Takes in a tensor with format (N, C, H, W), [0, 1] and returns a tensor of
# format (N, 1, H, W) [0, 1] that is 1 where the original images' red channel
# was 1 and 0 where the channel was less than 1.
def Threshold(images):
    # Input images should be in RGB [0, 1] format.
    assert images.size(1) == 3
    assert images.max() <= 1
    assert images.min() >= 0

    return (images[:, 0:1, :, :] == 1).double()
