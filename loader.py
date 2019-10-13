# Exports file location constants
import labeler
import glob
import os
from PIL import Image

kDataDir = "long_playthrough"
kCrop = (27, 398, 80, 417)

def GetAllFilenames():
    return glob.glob(os.path.join(kDataDir, "*.png"))

def LoadImages(labeled_images, crop = None):
    for label in labeled_images:
        # Ignore images that don't have scores present. Also note that we use
        # the text string "None" and not python's None type
        if label["money"] == "None":
            continue

        file = label["file"]
        # Image.open(file).crop(kCrop).show()
        # exit()
        labeler.DisplayImage(file, kCrop)
        input()
        # if crop is not None:

LoadImages(labeler.GetLabels())
