import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

kDataDir = "long_playthrough"
kLabelingFile = "labels.txt"

def GetAllFilenames():
    return glob.glob(os.path.join(kDataDir, "*.png"))

def GetAlreadyLabeledFilenames():
    filenames = []

    try:
        for line in open(kLabelingFile, "r").read():
            filenames.append(line.split()[0])
    except:
        pass

    return filenames

def DisplayImage(image_filename):
    plt.imshow(mpimg.imread(image_filename))
    plt.draw()
    plt.pause(.001)

def GetAnnotation():
    money = input("Money: ")
    tech_points = input("Tech Points: ")
    return money, tech_points

def WriteAnnotationToFile(image_filename, money, tech_points, labels_filename):
    with open(labels_filename, "a") as f:
        f.write(image_filename + " money " + str(money) + " tech_points " + str(tech_points) + "\n")

def LabelImages():
    to_label = set(GetAllFilenames())
    print("To label count: ", len(to_label))
    already_labeled = set(GetAlreadyLabeledFilenames())
    print("Already labeled count: ", len(already_labeled))
    to_label -= already_labeled
    while len(to_label) > 0:
        image_filename = to_label.pop()
        DisplayImage(image_filename)
        money, tech_points = GetAnnotation()
        WriteAnnotationToFile(image_filename, money, tech_points, kLabelingFile)

LabelImages()
