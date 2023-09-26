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
                for xl_idx in f[t].keys():
                    data_entry = {'t': t, 'xl_idx': xl_idx}
                    self.data.append(data_entry)

                    

       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as f:
            data_entry = self.data[index]
            t, xl_idx = data_entry['t'], data_entry['xl_idx']
            
            encoder_groups = [f[t][xl_idx]['encoder'][f'sample{sample_idx}'] for sample_idx in range(4)]
            decoder_groups = f[t][xl_idx]['decoder']['sample4']

            encoder_T_data = [g['T'][: , : ] for g in encoder_groups]
            encoder_vector_data = [g['vector'][ : ] for g in encoder_groups]
            decoder_vector_data = decoder_groups['vector']
            target_data = decoder_groups['T']

            encoder_T_tensor = torch.tensor(encoder_T_data, dtype=torch.float32)
            encoder_vector_tensor = torch.tensor(encoder_vector_data, dtype=torch.float32)
            decoder_vector_tensor = torch.tensor(decoder_vector_data, dtype=torch.float32)
            target_tensor = torch.tensor(target_data, dtype=torch.float32)

        return encoder_T_tensor, encoder_vector_tensor, decoder_vector_tensor, target_tensor

 






       
    

