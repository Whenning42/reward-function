# Save test image to test.png
# Write a label for test.png
# Load the image unlabled
# Load the image with a label

import pytest
import PIL.Image as Image
import loader
import numpy as np

def test_DataLoaderSingleImage(fs):
    kImagePath = "test.png"
    kLabelPath = "labels.txt"
    kDataFolder = "./"
    kImageCrop = (0, 0, 1, 1) # (left, upper, right, lower)

    image = Image.new("RGB", (3, 2)) # (width, height)
    image.save(kImagePath)

    data_loader = loader.DataLoader(kDataFolder, kLabelPath, kImageCrop)
    unlabeled_image = data_loader.LoadUnlabeledImages()[0]

    label_writer = loader.LabelWriter(kLabelPath)
    label_writer.WriteLabel(kImagePath, 10, 20)

    labeled_image = data_loader.LoadLabeledImages()[0]

    np.testing.assert_array_equal(unlabeled_image["image"], \
                                  labeled_image["image"])

    # Test that we can load labels
    assert unlabeled_image["label"] is None

    # Test that we store missing labels as None
    assert labeled_image["label"] is not None

    # Test that the crop worked
    assert labeled_image["image"].shape == (1, 1, 3)

def test_DataLoaderMultipleImages(fs):
    kLabeledImage = "labeled.png"
    kUnlabeledImage = "unlabeled.png"
    kLabelPath = "labels.txt"
    kDataFolder = "./"

    Image.new("RGB", (1, 1)).save(kLabeledImage)
    Image.new("RGB", (2, 2)).save(kUnlabeledImage)

    label_writer = loader.LabelWriter(kLabelPath)
    label_writer.WriteLabel(kLabeledImage, 0, 0)

    data_loader = loader.DataLoader(kDataFolder, kLabelPath)
    labeled_images = data_loader.LoadLabeledImages()
    unlabeled_images = data_loader.LoadUnlabeledImages()

    print("All", data_loader._GetAllImageFilenames())
    print("Unlabeled", data_loader._GetUnlabeledImageFilenames())
    print("Labeled", data_loader._GetLabeledImageFilenames())

    # Test that LoadLabeledImages gave us our labeled image
    assert labeled_images[0]["image"].shape == (1, 1, 3)
    # Test that LoadUnlabedImages gave us our unlabeled image
    assert unlabeled_images[0]["image"].shape == (2, 2, 3)
