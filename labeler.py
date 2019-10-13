import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

kLabelingFile = "labels.txt"

def GetAlreadyLabeledFilenames():
    filenames = []

    with open(kLabelingFile, "r") as f:
        for line in f.readlines():
            filenames.append(line.split()[0] + " " + line.split()[1])

    return filenames

def Sharpen(image):
    image[:, :, 1] = image[:, :, 0]
    image[:, :, 2] = image[:, :, 0]
    image = (image > .89) * 255
    return image

# We increase the recursion depth here because the recursive
# connected component call labels every pixel when called on
# an all white image. This gives a call depth of 19 * 53 which
# exceeds pythons default recursion depth.
import sys
sys.setrecursionlimit(2000)

kUnlabeledColor = (255, 255, 255)
kDefaultColor = (128, 128, 128)
def Sweep(image):
    current_component = 1
    for x in range(0, 50):
        if np.all(image[10, x] == kUnlabeledColor):
            Label(image, x, 10, int(255 / 5 * current_component))
            current_component += 1
    return image

def InBounds(image, x, y):
    return x >= 0 and  x < image.shape[1] and y >= 0 and y < image.shape[0]

def Label(image, x, y, color):
    if not InBounds(image, x, y):
        return

    image[y, x] = color
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if not InBounds(image, x + dx, y + dy):
                continue

            if np.all(image[y + dy, x + dx] == kUnlabeledColor):
                Label(image, x + dx, y + dy, color)

# This function can be called repeatedly to keep updating the displayed image
# as opposed to Image.show() which breaks in that case.
def DisplayImage(image_filename, crop = None):
    image = mpimg.imread(image_filename)
    if crop is not None:
        image = image[crop[1]:crop[3], crop[0]:crop[2]]
    image = Sharpen(image)
    image = Sweep(image)
    plt.imshow(image)
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
    already_labeled = set(GetAlreadyLabeledFilenames())
    to_label -= already_labeled

    print("To label count: ", len(to_label))
    print("Already labeled count:", len(already_labeled))

    while len(to_label) > 0:
        image_filename = to_label.pop()
        DisplayImage(image_filename)
        money, tech_points = GetAnnotation()
        WriteAnnotationToFile(image_filename, money, tech_points, kLabelingFile)

if __name__ == "__main__":
    LabelImages()
