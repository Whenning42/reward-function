import data_loader

class Config:
    def __init__(self):
        self.DataFolder = "long_playthrough"
        self.LabelFile = "labels.txt"

        # (x0, y0, x1, y1)
        self.Crop = (27, 398, 80, 417)

config = Config()

if __name__ == "__main__":
    label_filter = lambda label: label is not None \
                             and label["money"] not in ["None", "ILL"]
    loader = data_loader.DataLoader(config.DataFolder, \
                                    config.LabelFile, \
                                    crop = config.Crop, \
                                    label_filter = label_filter)

    images, labels, filenames = loader.LoadLabeledImages()

    model = MoneyClassifierModel()
    model.Train(image, labels, filenames)
    model.Evaluate
    train_end = int(len(dataset) / 7 * 10)
    train_set = dataset[:train_end]
    test_set = dataset[train_end:]

    model.train(train_set)
    model.evaluate(test_set)
