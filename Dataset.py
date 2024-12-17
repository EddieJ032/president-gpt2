from torch.utils.data import Dataset

class PresidentDataset(Dataset):
    """
    
    Args:
        data (list): The data tensors.

    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    # returns (input, output) tuple
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]