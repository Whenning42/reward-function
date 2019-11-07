# Takes ownership of the data passed into the constructor. The data used to
# constrcut this class shouldn't be used elsewhere.
class Dataset:
    def __init__(self, dataset):
        # Dataset should have an images tensor list, a labels list, and a UIDs
        # list.
        assert len(dataset) == 3
        for dim in range(len(dataset) - 1):
            assert len(dataset[dim]) == len(dataset[dim + 1])
        self.dataset = dataset

    def __getitem__(self, key):
        return Dataset(self, (sublist[key] for sublist in self.dataset))

    def get_images(self):
        return self.dataset[0]

    def get_labels(self):
        return self.dataset[1]

    def get_uids(self):
        return self.dataset[2]
