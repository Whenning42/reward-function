# Exports file location constants
import util
import glob
import os
import processing
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import itertools

kDataDir = "long_playthrough"
kLabelingFile = "labels.txt"
kCrop = (27, 398, 80, 417)
kCropDimensions = (kCrop[3] - kCrop[1], kCrop[2] - kCrop[0])

# If the labels change after class creation, this class will reflect
# that change.
kPathKey = "file"
class DataLoader:

    def __init__(self, data_folder, label_file, crop = None):
        self.crop = crop
        self.data_folder = data_folder
        self.label_file = label_file

    def _GetAllImageFilenames(self):
        files = glob.glob(os.path.join(self.data_folder, "*.png"))
        return [os.path.basename(file) for file in files]

    def _GetLabeledImageFilenames(self):
        values = []
        for label in self._LoadLabels():
            values.append(label[kPathKey])
        return values

    def _GetUnlabeledImageFilenames(self):
        return list(set(self._GetAllImageFilenames()) - set(self._GetLabeledImageFilenames()))

    # Returns a list of dictionaries holding the labels for every image
    # i.e. [{"file": "image.png", "money": 1025, "tech_points": 678}, ...]
    def _LoadLabels(self):
        labels = []
        if not os.path.exists(self.label_file):
            return labels
        with open(self.label_file, "r") as f:
            for line in f.readlines():
                fields = line.split()
                label = {kPathKey: fields[0]}
                for i in range(1, len(fields), 2):
                    label[fields[i]] = fields[i + 1]
                labels.append(label)
        return labels

    # Returns a list of an np array per image of shape (H, W, C = 3).
    # If crop is not None then the given crop will be applied to each image.
    # The crop should be of the form (x0, y0, x1, y2).
    def _LoadImages(self, filenames):
        images = []
        for i, filename in enumerate(filenames):
            image = Image.open(filename)
            if self.crop is not None:
                image = image.crop(self.crop)
            images.append(np.asarray(image))
        return images

    def _AddLabelsToImages(self, images, labels):
        labeled_images = []
        for image, label in zip(images, labels):
            labeled_images.append({"image": image, "label": label})
        return labeled_images

    ## Public

    # Loads from the set of images missing labels.
    # Returned image objects are of the form
    # {"image": image, "label", None}
    def LoadUnlabeledImages(self):
        images = self._LoadImages(self._GetUnlabeledImageFilenames())
        return self._AddLabelsToImages(images, itertools.repeat(None))


    # Loads from the set of images with labels.
    # Returned image objects are of the form
    # {"image": image, "label", label (just money for now)}
    def LoadLabeledImages(self):
        images = self._LoadImages(self._GetLabeledImageFilenames())
        labels = self._LoadLabels()
        return self._AddLabelsToImages(images, labels)

class LabelWriter:
    def __init__(self, label_file_path):
        self.label_file_path = label_file_path

    def WriteLabel(self, image_path, money, tech_points):
        with open(self.label_file_path, "a") as f:
            f.write(image_path + " money " + str(money) + " tech_points " + str(tech_points) + "\n")

# Also note that we use the text string "None" and not python's None type
def FilterOutImagesWithoutLabels(labels):
    filtered_labels = []
    for label in labels:
        if label["money"] != "None":
            filtered_labels.append(label)
    return filtered_labels

def SegmentImages(images):
    templates = []
    print("Segmenting", len(images), "images")
    for image in tqdm(images):
        _, templates_for_image = processing.GetConnectedComponents(image)
        for template in templates_for_image:
            # We use this format so that our labeled and unlabeled images can
            # have a common data structure.
            templates.append({"class": -1, "template": template})
    return templates

def SegmentImagesAndApplyLabels(templates, labels):
    labeled_templates = []
    for image, label in zip(tqdm(images), labels):
        component_vis, templates = processing.GetConnectedComponents(image)
        if len(label["money"]) != len(templates):
            print("Potential failure in connected components processing")
            print(label["money"])
            util.DisplayImage(component_vis, blocking = False)
            continue
        for i, template in enumerate(templates):
            labeled_templates.append({"class": int(label["money"][i]), "template": template})
    return labeled_templates

def TemplateIsOutlierSized(template):
    width = template.shape[1]
    height = template.shape[0]
    if height <= 11 or height >= 18:
        print("Unusual height:", height)
        print("Template:")
        util.DisplayImage(template * 255)
        return True
    if width >= 10:
        print("Unusual width:", width)
        print("Template:")
        util.DisplayImage(template * 255)
        return True
    return False


def FilterOutlierSizedTemplates(labeled_templates):
    for v in labeled_templates:
        width = v["template"].shape[1]
        height = v["template"].shape[0]
        if height <= 11 or height >= 18:
            print("Unusual height:", height)
            print("Class: ", v["class"])
            print("Template:")
            util.DisplayImage(v["template"] * 255)
            labeled_templates.remove(v)
            continue
        if width >= 10:
            print("Unusual width:", width)
            print("Class: ", v["class"])
            print("Template:")
            util.DisplayImage(v["template"] * 255)
            labeled_templates.remove(v)
            continue

def DisplayTemplateInfo(labeled_templates):
    template_widths = []
    template_heights = []
    classes = np.zeros((10,))

    for v in labeled_templates:
        width = v["template"].shape[1]
        height = v["template"].shape[0]
        template_widths.append(width)
        template_heights.append(height)
        classes[v["class"]] += 1

    for i in range(10):
        print(i, classes[i])

    # If we've possibly called util.DisplayImage, then we need to close the plot it created
    plt.close()

    plt.hist(template_widths)
    plt.title("Width distribution")
    plt.show()

    plt.hist(template_heights)
    plt.title("Height distribution")
    plt.show()

    util.ShowPrebinnedHistogram(classes, "Class distribution")

if __name__ == "__main__":
    print("Files remain unlabeled", len(GetUnlabeledFilenames()))
    # labels = LoadLabels()
    # labels = FilterOutImagesWithoutLabels(labels)

    # images = LoadImages(GetAlreadyLabeledFilenames())
    # labeled_templates = SegmentImagesAndApplyLabels(images, labels)
    unlabled_images = LoadImages(random.sample(GetUnlabeledFilenames(), 1000))
    unlabled_templates = SegmentImages(unlabled_images)

    FilterOutlierSizedTemplates(unlabled_templates)
    DisplayTemplateInfo(unlabled_templates)
