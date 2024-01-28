from torch.utils.data import Dataset

class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Load an image and apply transformations
        image = self.hf_dataset[idx]['image']
        label = self.hf_dataset[idx]['label']

        if self.transform:
            image = self.transform(image)
        return image, label