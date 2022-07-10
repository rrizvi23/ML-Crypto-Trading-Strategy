import torch

class SlidingDataset(torch.utils.data.Dataset):
  def __init__(self, sequence, targets, inp_dim):
        self.sequence = sequence
        self.targets = targets
        self.inp_dim = inp_dim

  def __len__(self):
        return len(self.targets) - self.inp_dim

  def __getitem__(self, index):
        y = self.targets[index + self.inp_dim]
        X = self.sequence[index : index + self.inp_dim]
        return X, y

class ClumpedDataset(torch.utils.data.Dataset):
  def __init__(self, sequence, targets, inp_dim):
        self.sequence = sequence
        self.targets = targets
        self.inp_dim = inp_dim        

  def __len__(self):
        return len(self.targets) // self.inp_dim

  def __getitem__(self, index):
        y = self.targets[index * self.inp_dim : (index + 1) * self.inp_dim]
        X = self.sequence[index * self.inp_dim : (index + 1) * self.inp_dim]
        return X, y