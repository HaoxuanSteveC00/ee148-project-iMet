from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class TrainDataset(Dataset):
    """iMet challenge dataset."""

    def __init__(self, train, label_info_csv, df_csv, root_dir, transform=None):
        """
        Args:
            label_info_csv (string): Path to the csv file with information about labels.
            target_csv (string): Path to the csv file with attribute ids.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.df = pd.read_csv(df_csv)
        self.labels_frame = pd.read_csv(label_info_csv)
        self.root_dir = root_dir
        self.transform = transform
        self.kmeans = np.loadtxt("data/kmeans20.txt", dtype=int)

        # self.labels = ['0', '1', '10', '100', '1000', '1001', '1002', '1003', '1004', '1005']
        # # self.label_idx = [int(t) for t in self.find_most_freq_labels(50)]
        # # print(self.label_idx)
        # # print(self.labels_frame.iloc[self.label_idx,:])
        #
        # # self.df = self.preprocess()


    # def find_most_freq_labels(self, n):
    #     """
    #     Find the top n most frequent attributes. Return a list of attribute ids.
    #     """
    #     attribute_ids = self.df.iloc[:,1].str.split()
    #     attribute_ids = pd.Series(np.concatenate(attribute_ids))
    #     most_freq_labels = attribute_ids.value_counts().sort_index().rename_axis('x').reset_index(name='f')['x'].iloc[0:n].tolist()
    #     return most_freq_labels


    # def preprocess(self):
    #     df = pd.DataFrame([])
    #     for index, row in self.df.iterrows():
    #         atts = row['attribute_ids'].split()
    #         for l in self.labels:
    #             if l in atts:
    #                 df = df.append(self.df.iloc[index,:])
    #                 df.iloc[-1,1] = l
    #                 break
    #     # print(df.shape)
    #     return df


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.df['id'].values[idx]
        file_path = f'./data/imet-2020-fgvc7/train/{file_name}.png'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        if not self.train:
            return image

        # target = torch.zeros(N_CLASSES)
        # for cls in self.df.iloc[idx].attribute_ids.split():
        #     target[int(cls)] = 1

        # target = torch.tensor(int(self.df.iloc[idx].attribute_ids.split()[0]), dtype=torch.long)

        target = torch.tensor(self.kmeans[idx], dtype=torch.long)
        return image, target



# t = iMetDataset(True, 'data/imet-2020-fgvc/labels.csv', 'data/imet-2020-fgvc/train.csv', 'data/imet-2020-fgvc/train')
