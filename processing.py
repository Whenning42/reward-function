import numpy as np

def ThresholdImage(image):
    image[:, :, 1] = image[:, :, 0]
    image[:, :, 2] = image[:, :, 0]
    image = (image == 255) * 255
    return image

def InBounds(image, pos):
    return pos[0] >= 0 and pos[0] < image.shape[0] and pos[1] >= 0 and pos[1] < image.shape[1]

# We increase the recursion depth here because the recursive
# connected component call labels every pixel when called on
# an all white image. This gives a call depth of 19 * 53 which
# exceeds pythons default recursion depth.
import sys
sys.setrecursionlimit(5000)
def LabelComponentRecursively(image, current_position, color_to_label, component):
    image[current_position] = color_to_label
    component[current_position] = 1
    for d0 in range(-1, 2):
        for d1 in range(-1, 2):
            next_pos = (current_position[0] + d0, current_position[1] + d1)
            if not InBounds(image, next_pos):
                continue
            if np.all(image[next_pos] == kUnlabeledComponent):
                LabelComponentRecursively(image, next_pos, color_to_label, component)

# Returns an numpy array of size matching the extent of the component
# holding 0-1 pixel dense labeling of the component
def LabelComponent(image, current_position, color_to_label):
    template = np.zeros(image.shape)
    LabelComponentRecursively(image, current_position, color_to_label, template)

    nonzero_indices = np.where(template > 0)
    extent = np.min(nonzero_indices[0]), \
             np.max(nonzero_indices[0]), \
             np.min(nonzero_indices[1]), \
             np.max(nonzero_indices[1])
    return template[extent[0] : extent[1] + 1, extent[2] : extent[3] + 1]

import util

# image here is an np array.
kUnlabeledComponent = (255, 255, 255)
def GetConnectedComponents(image):
    kMaxComponents = 6

    working_image = image.copy()
    working_image = ThresholdImage(working_image)
    components = []
    current_component = 0
    kYSweepPos = 10
    for x in range(0, image.shape[1]):
        pos = (kYSweepPos, x)
        if np.all(working_image[pos] == kUnlabeledComponent):
            # Here we ensure that label_color is never set to kUnlabledComponent
            # which is 255.
            if current_component == kMaxComponents - 1:
                print("Encountered more components than expected")
                current_component = 0
            label_color = ((current_component + 1) / kMaxComponents * 255)
            components.append(LabelComponent(working_image, pos, label_color))
            current_component += 1

    return working_image, components
