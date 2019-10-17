import matplotlib.pyplot as plt
import numpy as np

# This function can be called repeatedly to keep updating the displayed image
# as opposed to PIL's Image.show() which breaks in that case.
def DisplayImage(image):
    assert(isinstance(image, (np.ndarray, np.generic))), \
            "DisplayImage was passed something that isn't an np array"
    image = image / 255
    plt.imshow(image)
    plt.draw()
    plt.pause(.001)
    input()
