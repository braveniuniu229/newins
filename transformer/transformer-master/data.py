"""
@author : zwz
@when : 2023-9-5
@homepage : https://github.com/braveniuniu229
"""
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomHDF5Dataset(Dataset):
    def __init__(self, file_path, train=True):
        super(CustomHDF5Dataset, self).__init__()
        
        self.file_path = file_path
        self.train = train
        self.data = []

        self._load_data()

    def _load_data(self):
        with h5py.File(self.file_path, 'r') as f:
            for t in f.keys():
                for p in f[t].keys():
                    groups = [f[t][p][f'group_{i}'] for i in range(40)]
                    
                    for i in range(0, 40, 5):
                        group_set = groups[i:i+5]

                        # Extract encoder and decoder data
                        encoder_data = [g['key'][:] for g in group_set[:-1]]
                        decoder_data = group_set[-1]['query'][:]

                        data_entry = {
                            "encoder": encoder_data,
                            "decoder": decoder_data
                        }

                        # If it's the last set, append to validation set if required
                        if i == 35:
                            if not self.train:
                                self.data.append(data_entry)
                        else:
                            if self.train:
                                self.data.append(data_entry)

        # Shuffle the data
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_entry = self.data[index]

        encoder_tensor = torch.tensor(data_entry['encoder'], dtype=torch.float32)
        decoder_tensor = torch.tensor(data_entry['decoder'], dtype=torch.float32)

        return encoder_tensor, decoder_tensor
 
# Usage:
train_dataset = CustomHDF5Dataset('your_dataset_path.hdf5', train=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomHDF5Dataset('your_dataset_path.hdf5', train=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


       
    

