
from torch.utils.data import DataLoader

def get_dataloaders(
    data_root: str,
    split_dir: str,
    batch_size: int = 8,
    num_workers: int = 2,
    use_mask: bool = False
):
    dataset_cls = (
        __import__('src.datasetmask', fromlist=['PairedMultiViewMaskDataset']).PairedMultiViewMaskDataset 
        if use_mask else
        __import__('src.datasetnomask', fromlist=['PairedMultiViewDataset']).PairedMultiViewDataset
    )

    train_dataset = dataset_cls(data_root, f"{split_dir}/train.txt")
    val_dataset   = dataset_cls(data_root, f"{split_dir}/val.txt")
    test_dataset  = dataset_cls(data_root, f"{split_dir}/test.txt")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
