import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import loader

def PromptUserForAnnotation():
    money = input("Money: ")
    tech_points = input("Tech Points: ")
    return money, tech_points

# TODO move to loader
def WriteAnnotationToFile(image_filename, money, tech_points, labels_filename):
    with open(labels_filename, "a") as f:
        f.write(image_filename + " money " + str(money) + " tech_points " + str(tech_points) + "\n")

def LabelImages():
    to_label = set(loader.GetAllFilenames())
    already_labeled = set(loader.GetAlreadyLabeledFilenames())
    to_label -= already_labeled

    print("To label count: ", len(to_label))
    print("Already labeled count:", len(already_labeled))

    while len(to_label) > 0:
        image_filename = to_label.pop()
        DisplayImage(image_filename)
        money, tech_points = PromptUserForAnnotation()
        WriteAnnotationToFile(image_filename, money, tech_points, kLabelingFile)

if __name__ == "__main__":
    LabelImages()
