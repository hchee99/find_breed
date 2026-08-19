import csv
import os
import random

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class DogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_root = root_dir
        self.transform = transform
        self.labels = []
        self.image_path = []

        self.classes = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d != "annotations"
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        valid_extensions = ('.jpg', '.jpeg', '.png')
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for file_name in sorted(os.listdir(cls_dir)):    
                if file_name.lower().endswith(valid_extensions):
                    self.image_path.append(os.path.join(cls_dir, file_name))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index]).convert('RGB')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

def stratified_split(dataset, ratios=(0.7, 0.15, 0.15), seed=42):
    by_class = {}
    for i, label in enumerate(dataset.labels):
        by_class.setdefault(label, []).append(i)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label in sorted(by_class):
        idx = by_class[label][:]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train_idx += idx[:n_train]
        val_idx += idx[n_train:n_train + n_val]
        test_idx += idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def save_splits(dataset, splits, out_path='splits.csv'):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path', 'breed', 'split'])
        for name, idx_list in zip(('train', 'val', 'test'), splits):
            for i in idx_list:
                w.writerow([dataset.image_path[i],
                            dataset.classes[dataset.labels[i]], name])

def get_dataloader(root_dir, batch_size=8):
    trans = transforms.Compose([
        transforms.Resize((518, 518)),                       
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),          
                             (0.229, 0.224, 0.225)),
    ])
    data = DogDataset(root_dir, transform=trans)
    tr, va, te = stratified_split(data)
    save_splits(data, (tr, va, te))

    return (DataLoader(Subset(data, tr), batch_size=batch_size, shuffle=True),
            DataLoader(Subset(data, va), batch_size=batch_size),
            DataLoader(Subset(data, te), batch_size=batch_size))
