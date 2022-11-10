class NI4OSResult():
    def __init__(self, image=None, mimetype=None, results=None, ood=None):
        self.image = image
        self.mimetype = mimetype
        self.results = results
        self.ood = ood


class NI4OSData():
    def __init__(self):
        self.data = []
        self.mimetype = []
        self.results = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.mimetype[idx], self.results[idx]
