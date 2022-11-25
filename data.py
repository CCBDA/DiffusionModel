from torch.utils.data import Dataset
import os
from PIL import Image


class DiffusionSet(Dataset):
    def __init__(self, tfs=None, root_dir="mnist") -> None:
        self.files = os.listdir(root_dir)
        self.root_dir = root_dir
        self.tfs = tfs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.files[index])
        image = Image.open(path)
        if self.tfs is not None:
            image = self.tfs(image)
        return image
