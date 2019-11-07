# Save test image to test.png
# Write a label for test.png
# Load the image unlabled
# Load the image with a label

import pytest
import PIL.Image as Image
import data_loader
import numpy as np

def test_DataLoaderSingleImage(fs):
    kImagePath = "test.png"
    kLabelPath = "labels.txt"
    kDataFolder = "./"
    kImageCrop = (0, 0, 40, 20) # (x0, y0, x1, y1)
    kMoney = 10
    kTechPoints = 20

    image = Image.new("RGB", (80, 60)) # Note the format here is (W, H) which
                                       # is the opposite of torch.
    image.save(kImagePath)

    # Create the label file
    with open(kLabelPath, 'w+'): pass

    loader = data_loader.DataLoader(kDataFolder, kLabelPath, kImageCrop)
    unlabeled_images, _, _ = loader.LoadUnlabeledImages()
    unlabeled_image = unlabeled_images[0]

    label_writer = data_loader.LabelWriter(kLabelPath)
    label_writer.WriteLabel(kImagePath, kMoney, kTechPoints)

    labeled_images, labels, labeled_filenames = loader.LoadLabeledImages()
    label = labels[0]
    labeled_image = labeled_images[0]

    np.testing.assert_array_equal(unlabeled_image, \
                                  labeled_image)

    # Test that we can load labels
    assert int(label["money"]) == kMoney
    assert int(label["tech_points"]) == kTechPoints

    # Test that the crop worked
    assert labeled_image.shape == (3, 20, 40) # (C, H, W)

def test_DataLoaderMultipleImages(fs):
    kLabeledImage = "labeled.png"
    kUnlabeledImage = "unlabeled.png"
    kLabelPath = "labels.txt"
    kDataFolder = "./"

    images_dim = (10, 5) # (H, W)

    Image.new("RGB", images_dim[::-1]).save(kLabeledImage) # (W, H)
    Image.new("RGB", images_dim[::-1]).save(kUnlabeledImage) # (W, H)

    label_writer = data_loader.LabelWriter(kLabelPath)
    label_writer.WriteLabel(kLabeledImage, 0, 0)

    loader = data_loader.DataLoader(kDataFolder, kLabelPath)
    labeled_images, _, _ = loader.LoadLabeledImages()
    unlabeled_images, _, _ = loader.LoadUnlabeledImages()

    # Test that LoadLabeledImages gave us our labeled image
    assert labeled_images[0].shape == (3, *images_dim)
    # Test that LoadUnlabedImages gave us our unlabeled image
    assert unlabeled_images[0].shape == (3, *images_dim)
