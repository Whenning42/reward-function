import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

kLabelingFile = "labels.txt"

def GetLabels():
    labels = []
    with open(kLabelingFile, "r") as f:
        for line in f.readlines():
            fields = line.split()
            label = {"file": fields[0]}
            for i in range(1, len(fields), 2):
                label[fields[i]] = fields[i + 1]
            labels.append(label)
    return labels

def GetFieldFromLabels(field):
    values = []
    for label in GetLabels():
        values.append(label[field])
    return values

def GetAlreadyLabeledFilenames():
    return GetFieldFromLabels("money")

def PromptUserForAnnotation():
    money = input("Money: ")
    tech_points = input("Tech Points: ")
    return money, tech_points

def WriteAnnotationToFile(image_filename, money, tech_points, labels_filename):
    with open(labels_filename, "a") as f:
        f.write(image_filename + " money " + str(money) + " tech_points " + str(tech_points) + "\n")

def LabelImages():
    to_label = set(GetAllFilenames())
    already_labeled = set(GetAlreadyLabeledFilenames())
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
