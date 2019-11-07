if __name__ == "__main__":
    loader = DataLoader(kDataFolder, kLabelFile, kCrop)
    dataset = loader.LoadLabeledImages()
    model = SkyrogueModel()

    train_end = int(len(dataset) / 7 * 10)
    train_set = dataset[:train_end]
    test_set = dataset[train_end:]

    model.train(train_set)
    model.evaluate(test_set)
