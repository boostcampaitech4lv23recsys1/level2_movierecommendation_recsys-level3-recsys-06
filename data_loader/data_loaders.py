from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import DataLoader
from .context_data_loader import StaticDataset, StaticTestDataset
from .sequential_data_loader import SeqTrainDataset, SeqTestDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class FMDataLoader(DataLoader):
    """
    FM data loading demo using BaseDataLoader
    """
    def __init__(self, dataset, config, batch_size, shuffle, num_workers=4):
        
        self.init_kwargs = {
            'dataset' : dataset,
            'batch_size': batch_size // (1 + config['neg_ratio']) if isinstance(dataset, StaticDataset) else 4096,
            'shuffle': isinstance(dataset, StaticDataset),
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)


class SeqDataLoader(DataLoader):
    """
    FM data loading demo using BaseDataLoader
    """
    def __init__(self, dataset, config, batch_size, shuffle, num_workers=4):
        
        self.init_kwargs = {
            'dataset' : dataset,
            'batch_size': batch_size,
            'shuffle': isinstance(dataset, SeqTrainDataset),
            'num_workers': num_workers
        }
         
        super().__init__(**self.init_kwargs)