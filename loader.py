# Exports file location constants
import util
import labeler
import glob
import os
import processing
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np

kDataDir = "long_playthrough"
kCrop = (27, 398, 80, 417)
kCropDimensions = (kCrop[3] - kCrop[1], kCrop[2] - kCrop[0])

def GetAllFilenames():
    return glob.glob(os.path.join(kDataDir, "*.png"))

def LoadImage(filename, crop = None):
    image = Image.open(filename)
    if crop is not None:
        image = image.crop(crop)
    return np.asarray(image)

# Also note that we use the text string "None" and not python's None type
def FilterOutImagesWithoutLabels(labels):
    filtered_labels = []
    for label in labels:
        if label["money"] != "None":
            filtered_labels.append(label)
    return filtered_labels

def LoadImages(labels, crop = None):
    num_images = len(labels)
    images = np.empty((num_images, *kCropDimensions, 3))

    for i, label in enumerate(labels):
        images[i] = LoadImage(label["file"], kCrop)
    return images

def GetTemplatesFromImages(images, labels):
    labeled_templates = []
    for image, label in zip(tqdm(images), labels):
        component_vis, templates = processing.GetConnectedComponents(image)
        if len(label["money"]) != len(templates):
            print("Potential failure in connected components processing")
            print(label["money"])
            util.DisplayImage(component_vis)
            continue
        for i, template in enumerate(templates):
            labeled_templates.append({"class": int(label["money"][i]), "template": template})
    return labeled_templates

labels = labeler.GetLabels()
labels = FilterOutImagesWithoutLabels(labels)

images = LoadImages(labels)
labeled_templates = GetTemplatesFromImages(images, labels)

template_widths = []
template_heights = []
# Using 11 here is a workaround for the confusing behavior of plt.hist()
classes = np.zeros((11,))

for v in labeled_templates:
    width = v["template"].shape[1]
    height = v["template"].shape[0]
    if height <= 11 or height >= 18:
        print("Unusual height:", height)
        print("Class: ", v["class"])
        print("Template:")
#        util.DisplayImage(v["template"] * 255)
        labeled_templates.remove(v)
        continue
    if width >= 10:
        print("Unusual width:", width)
        print("Class: ", v["class"])
        print("Template:")
#        util.DisplayImage(v["template"] * 255)
        labeled_templates.remove(v)
        continue
    template_widths.append(width)
    template_heights.append(height)
    classes[v["class"]] += 1

for i in range(10):
    print(i, classes[i])

# If we called util.DisplayImage, then we need to close the default plot
plt.close()
plt.hist(template_widths)
plt.title("Width distribution")
plt.show()
plt.hist(template_heights)
plt.title("Height distribution")
plt.show()
# Using 11 here is a workaround for the confusing behavior of plt.hist()
plt.hist(range(0, 11), bins = range(0, 11), weights = classes)
plt.title("Class distribution")
plt.show()
