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
import ml

kDataFolder = "long_playthrough"
kLabelFile = "labels.txt"
kCrop = (27, 398, 80, 417)
kMaxDigitSize = (16, 8, 3)
kNumClasses = 10
# kCropDimensions = (kCrop[3] - kCrop[1], kCrop[2] - kCrop[0])

# If the labels change after class creation, this class will reflect
# that change.
kPathKey = "file"
class DataLoader:

    def __init__(self, data_folder, label_file, crop = None):
        self.crop = crop
        self.data_folder = data_folder
        self.label_file = label_file

    def GetAllImageFilenames(self):
        files = glob.glob(os.path.join(self.data_folder, "*.png"))
        return [os.path.basename(file) for file in files]

    def GetLabeledImageFilenames(self):
        values = []
        for label in self._LoadLabels():
            values.append(label[kPathKey])
        return values

    def GetUnlabeledImageFilenames(self):
        return list(set(self.GetAllImageFilenames()) - set(self.GetLabeledImageFilenames()))

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
        for i, filename in enumerate(tqdm(filenames)):
            image = Image.open(os.path.join(kDataFolder, filename))
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
    def LoadUnlabeledImages(self, max_images = None):
        files = self.GetUnlabeledImageFilenames()
        if max_images is not None and max_images < len(files):
            files = random.sample(files, max_images)

        images = self._LoadImages(files)
        return self._AddLabelsToImages(images, itertools.repeat(None))

    # Loads from the set of images with labels.
    # Returned image objects are of the form
    # {"image": image, "label", label (just money for now)}
    def LoadLabeledImages(self):
        images = self._LoadImages(self.GetLabeledImageFilenames())
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
            # We use this format so that our labeled and unlabeled templates can
            # have a common data structure.
            templates.append({"class": -1, "template": template})
    return templates

def SegmentImagesAndApplyLabels(labeled_images):
    labeled_templates = []
    for labeled_image in tqdm(labeled_images):
        image = labeled_image["image"]
        label = labeled_image["label"]
        component_vis, templates = processing.GetConnectedComponents(image)

        if label is None:
            label = {"money": len(templates) * [-1]}

        if len(label["money"]) != len(templates):
            print("Number of connected components differs from the number of digit labels")
            print("Label is:", label["money"])
            print("Segmentation output:")
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
    def OutlierSized(labeled_template):
        template = labeled_template["template"]
        width = template.shape[1]
        height = template.shape[0]
        if height <= 11 or height > 16:
            print("Unusual height:", height)
            print("Class: ", labeled_template["class"])
            print("Template:")
            # util.DisplayImage(template * 255)
            return False
        if width > 8:
            print("Unusual width:", width)
            print("Class: ", labeled_template["class"])
            print("Template:")
            # util.DisplayImage(template * 255)
            return False
        return True

    # I know this might not be performant, but big dogs do what big dogs want.
    return list(filter(OutlierSized, labeled_templates))

def DisplayTemplateInfo(labeled_templates, show_class_distribution = True):
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

    if show_class_distribution:
        util.ShowPrebinnedHistogram(classes, "Class distribution")

# new_size is given as (height, width)
def ResizeTemplatesInPlace(labeled_templates, new_size):
    print("Resizing templates")
    for labeled_template in tqdm(labeled_templates):
        template = labeled_template["template"]
        old_height, old_width = template.shape[0:2]
        resized = np.zeros(new_size)
        offset_x = new_size[1] // 2 - old_width // 2
        offset_y = new_size[0] // 2 - old_height // 2
        resized[offset_y : offset_y + old_height, \
                offset_x : offset_x + old_width] = template
        labeled_template["template"] = resized

if __name__ == "__main__":
    loader = DataLoader(kDataFolder, kLabelFile, kCrop)
    print("Images remaining unlabeled", len(loader.GetUnlabeledImageFilenames()))
    labeled_images = loader.LoadLabeledImages()
    labeled_templates = SegmentImagesAndApplyLabels(labeled_images)
    labeled_templates = FilterOutlierSizedTemplates(labeled_templates)

    # DisplayTemplateInfo(labeled_templates, show_class_distribution = False)
    ResizeTemplatesInPlace(labeled_templates, kMaxDigitSize) # (height, width)

    num_templates = len(labeled_templates)
    train_templates = labeled_templates[: 5 * num_templates // 10]
    val_templates = labeled_templates[5 * num_templates // 10 : ]
    # test_templates = labeled_templates[8 * num_templates // 10 : ]

    train_x, train_y = ml.ConvertLabeledTemplatesToTensors(train_templates, kNumClasses)
    val_x, val_y = ml.ConvertLabeledTemplatesToTensors(val_templates, kNumClasses)
    # test_x, test_y = ml.ConvertLabeledTemplatesToTensors(test_templates, kNumClasses)

    classifier = ml.Dense2DClassifier(kNumClasses, kMaxDigitSize).cuda()
    ml.TrainClassifier(classifier, train_x, train_y, val_x, val_y)
    # EvaluateClassifier(classifier, test_x, test_y)

    unlabeled_images = loader.LoadUnlabeledImages(200)
    unlabeled_templates = SegmentImagesAndApplyLabels(unlabeled_images)
    unlabeled_templates = FilterOutlierSizedTemplates(unlabeled_templates)
    ResizeTemplatesInPlace(unlabeled_templates, kMaxDigitSize)

    test_x, _ = ml.ConvertLabeledTemplatesToTensors(unlabeled_templates, kNumClasses)
    for i in range(test_x.size(0)):
        print("Pictured test sample is classified as: ", classifier(test_x[i : i + 1]))
        util.DisplayImage(test_x[i].cpu().numpy() * 255)
