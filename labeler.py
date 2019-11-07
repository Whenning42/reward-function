import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import data_loader
import config

def PromptUserForAnnotation():
    money = input("Money: ")
    tech_points = input("Tech Points: ")
    return money, tech_points

# TODO move to loader
def WriteAnnotationToFile(image_filename, money, tech_points, labels_filename):
    with open(labels_filename, "a") as f:
        f.write(image_filename + " money " + str(money) + " tech_points " + str(tech_points) + "\n")

def LabelImages():
    config = config.Config()

    loader = data_loader.DataLoader(config.DataFolder, config.LabelFile)
    _, _, to_label_filenames = data_loader.LoadUnlabeledImages()
    to_label = set(to_label_filenames)

    print("To label count: ", len(to_label))
    print("Already labeled count:", len(already_labeled))

    # Randomness of samples is contingent on the behavior of set.pop(). (I \
    # haven't looked it up).
    while len(to_label) > 0:
        image_filename = to_label.pop()
        DisplayImage(image_filename)
        money, tech_points = PromptUserForAnnotation()
        WriteAnnotationToFile(image_filename, money, tech_points, kLabelingFile)

if __name__ == "__main__":
    LabelImages()
