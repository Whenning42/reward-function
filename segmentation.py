# Needs to support caching the segmentations and accessing them
# with a UID.

from collections import deque
import torch

def InBounds(image, pos):
    return pos[0] >= 0 and \
           pos[0] < image.shape[0] and \
           pos[1] >= 0 and \
           pos[1] < image.shape[1]


# This class uses torch batches of images with a (N, C, H, W) batch image layout.
class SkyrogueSegmenter:
    def __init__(self):
        self.kUnlabeledComponent = -1

    def Threshold(self, images):
        # Input images should be in RGB 0-1 format.
        assert images.size(1) == 3
        assert images.max() <= 1
        assert images.min() >= 0

        return (images[:, 0:1, :, :] == 1).double()

    @staticmethod
    def _ConnectComponent(working_image, \
                          start_position, \
                          component_label, \
                          component_bitmap, \
                          unlabeled_sentinel):
        to_connect = deque()
        to_connect.append(start_position)
        while len(to_connect) > 0:
            position = to_connect.popleft()
            for d0 in range(-1, 2):
                for d1 in range(-1, 2):
                    next_pos = (position[0] + d0, position[1] + d1)
                    if not InBounds(working_image, next_pos):
                        continue
                    if working_image[next_pos] == unlabeled_sentinel:
                        working_image[next_pos] = component_label
                        component_bitmap[next_pos] = 1
                        to_connect.append(next_pos)

    @staticmethod
    def _CropToExtent(bitmap):
        nonzero_indices = bitmap.nonzero()
        extent = nonzero_indices[:, 0].min(), \
                 nonzero_indices[:, 0].max(), \
                 nonzero_indices[:, 1].min(), \
                 nonzero_indices[:, 1].max()
        return bitmap[extent[0] : extent[1] + 1, \
                      extent[2] : extent[3] + 1]

    # Returns a list of lists of pytorch tensors cropped to the size of their
    # segmentations
    def Segment(self, images):
        assert images.size(1) == 1
        assert images.max() <= 1
        assert images.min() >= 0

        segmentations = []
        for image in images:
            components = []

            # Here image is (C, H, W) with C = 1 so we strip the channel here
            # to get working image into a (H, W) format.
            working_image = image[0] * self.kUnlabeledComponent

            y = images.size(2) // 2
            current_component = 1
            for x in range(0, images.size(3)):
                if working_image[y, x] == self.kUnlabeledComponent:
                    component_bitmap = torch.zeros(working_image.size())
                    self._ConnectComponent(working_image, \
                                           (y, x), \
                                           current_component, \
                                           component_bitmap, \
                                           self.kUnlabeledComponent)

                    components.append(self._CropToExtent(component_bitmap))
                    current_component += 1

            segmentations.append(components)
        return segmentations
