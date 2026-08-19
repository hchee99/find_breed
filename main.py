from Utils.DataLoader import DogDataset, stratified_split, save_splits

ROOT = r'C:\Users\user\Documents\GitHub\find_breed\dog_breed_dataset'

if __name__ == '__main__':
    data = DogDataset(ROOT)
    tr, va, te = stratified_split(data)
    save_splits(data, (tr, va, te))
