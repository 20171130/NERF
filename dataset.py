import torch
from torch.utils.data import Dataset
from random import shuffle
from torch.nn.utils.rnn import pad_sequence
import pdb


class TransformerDataset(Dataset):
    """
    src/tgt element, bond, charge, aroma, mask
    reactant, segment
    """
    def __init__(self, if_shuffle, data):
        # (L, B, C)
        self.data = data # list of dict of tensors
        self.shuffle = if_shuffle
        for feature_dict in data:
            for key in feature_dict:
                tmp = torch.tensor(feature_dict[key])
                if 'aroma' in key or 'mask' in key:
                    feature_dict[key] = tmp.bool()
                else:
                    feature_dict[key] = tmp.long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.shuffle:
            length = data['element'].shape[0]
            index = []
            index = list(range(length))
            reverse = [(x, y) for x, y in enumerate(index)]
            reverse.sort(key = lambda x: x[1])
            reverse = [i[0] for i in reverse]
            reverse = torch.tensor(reverse).unsqueeze(1).expand(-1, data['src_bond'].shape[1])
            for key in data:
                if 'bond' in key:
                    data[key] = data[key][index]
                    data[key] = torch.gather(reverse, 0, data[key])
                else:
                    data[key] = data[key][index]
        return data
    
    def collate_fn(data_list):
        # variable lenght batch, minimal padding
        max_len = 0
        for i in data_list:
            if i['element'].shape[0] > max_len:
                max_len = i['element'].shape[0]
        batch = {}
        for key in data_list[0]:
            lst = [i[key] for i in data_list]
            batch[key] = pad_sequence(lst, batch_first = True, padding_value = 1)
        return batch

