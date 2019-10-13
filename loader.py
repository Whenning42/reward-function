# Exports file location constants
import labeler
import glob
import os
from PIL import Image

kDataDir = "long_playthrough"
kCrop = (27, 398, 80, 417)

def GetAllFilenames():
    return glob.glob(os.path.join(kDataDir, "*.png"))

def LoadImages(filenames, crop = None):
    for file in filenames[10:]:
        # Image.open(file).crop(kCrop).show()
        # exit()
        labeler.DisplayImage(file, kCrop)
        input()
        # if crop is not None:

LoadImages(GetAllFilenames())
