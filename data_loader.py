import glob
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

class LabelWriter:
    def __init__(self, label_file_path):
        self.label_file_path = label_file_path

    def WriteLabel(self, image_path, money, tech_points):
        with open(self.label_file_path, "a") as f:
            f.write("file " + image_path + \
                    " money " + str(money) + \
                    " tech_points " + str(tech_points) + "\n")

# If the label file changes after class creation, this class will reflect
# those changes.
class DataLoader:
    # The dataloader will never load images that are labeled with a label for
    # which label_filter(label) = False. This is useful for filtering out
    # sample images that shouldn't be trained on.
    def __init__(self, data_folder, label_file, crop = None, label_filter = None):
        self.data_folder = data_folder
        self.label_file = label_file
        self.crop = crop
        if crop is not None:
            self.image_dimensions = (crop[3] - crop[1], crop[2] - crop[0])
        else:
            self.image_dimensions = None
        self.label_filter = label_filter

    def _LoadLabels(self, label_filter = None):
        assert self.label_file is not None
        assert os.path.exists(self.label_file)

        labels = []
        with open(self.label_file, "r") as f:
            for line in f.readlines():
                fields = line.split()
                label = {}
                for i in range(0, len(fields), 2):
                    # Add whatever keys and values are in the label file at
                    # this row.
                    label[fields[i]] = fields[i + 1]
                if self.label_filter is not None:
                    if not self.label_filter(label):
                        continue

                labels.append(label)
        return labels

    def _GetLabeledImageFilenames(self, label_filter = None):
        return [label["file"] for label in self._LoadLabels(label_filter)]

    def _GetUnlabeledImageFilenames(self):
        all_images = [os.path.basename(file) for file in \
                        glob.glob(os.path.join(self.data_folder, "*.png"))]
        labeled_images = self._GetLabeledImageFilenames()
        return list(set(all_images) - set(labeled_images))

    def _LoadImages(self, filenames):
        if self.image_dimensions is None and len(filenames) > 0:
            # Size is returned as (W, H) but we want this stored as (H, W)
            self.image_dimensions = Image.open( \
                os.path.join(self.data_folder, filenames[0])).size[::-1]

        images = torch.zeros(len(filenames), 3, *self.image_dimensions)
        for i, filename in enumerate(tqdm(filenames)):
            image = Image.open(os.path.join(self.data_folder, filename))
            if self.crop is not None:
                image = image.crop(self.crop)
            np_array = np.asarray(image)
            # Change from (H, W, C) format to (C, H, W) format
            np_array = np.transpose(np_array, (2, 0, 1))
            images[i] = torch.from_numpy(np_array).float() / 255
        return images

    # Returns images, labels, filenames where
    #   images is a tensor of format (N, C, H, W), [0, 1]
    #   labels are of format [{"file": "image.png",
    #                          "money": "1245",
    #                          "tech_points": "650"}] * N
    #   and filenames are of format ["image.png"] * N
    #
    # Exludes any labeled images for which label_filter(label) = False
    def LoadLabeledImages(self):
        filenames = self._GetLabeledImageFilenames(self.label_filter)
        images = self._LoadImages(filenames)
        labels = self._LoadLabels(filenames)
        return images, labels, filenames

    # Exludes any labeled images for which label_filter(label) = False
    def LoadUnlabeledImages(self):
        filenames = self._GetUnlabeledImageFilenames()
        images = self._LoadImages(filenames)
        labels = [{"file": filename, \
                   "money": "None", \
                   "tech_points": "None"} for filename in filenames]
        return images, labels, filenames
