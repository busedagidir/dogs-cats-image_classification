import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, root_dir, annotations, transform=None, target_transform=None):

        self.root_dir = root_dir # "E:\datasets\dogs-vs-cats"
        #self.csv_file()
        self.annotations = pd.read_csv(annotations, error_bad_lines=False) # dataset_csv fonk cagrilacak
        self.transform = transform


    def __len__(self):
        return len(self.annotations)

    # returns specific image and corresponding label
    def __getitem__(self, idx):
        #print("img_path_NE?:")
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        #print("img_path:",img_path)
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[idx, 2]))

        if self.transform:
            image = self.transform(image)

        return image, y_label








