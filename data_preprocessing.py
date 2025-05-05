# data_preprocessing.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class BearingDataset(Dataset):
    """
    IMS Bearing Dataset을 Torch Dataset 형식으로 변환하여 로드
    다운샘플링 및 채널 축소 포함
    """
    def __init__(self, data_dir, test_set='1st_test', transform=None, downsample_ratio=10):
        self.data_dir = os.path.join(data_dir, test_set)
        self.file_list = sorted(os.listdir(self.data_dir))
        self.transform = transform
        self.downsample_ratio = downsample_ratio
        self.num_channels = 8 if test_set == '1st_test' else 4

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = pd.read_csv(file_path, header=None, sep='\t').values

        if self.num_channels == 8:
            data = np.column_stack([
                (data[:, 0] + data[:, 1]) / 2,
                (data[:, 2] + data[:, 3]) / 2,
                (data[:, 4] + data[:, 5]) / 2,
                (data[:, 6] + data[:, 7]) / 2
            ])

        data = data[::self.downsample_ratio]  # 다운샘플링 적용

        if self.transform:
            data = self.transform(data)
        return data

class DataPreprocessor:
    """
    각 채널별로 독립적인 스케일링을 수행
    """
    def __init__(self, num_channels):
        self.scalers = [StandardScaler() for _ in range(num_channels)]

    def fit_transform(self, data):
        transformed_data = np.zeros_like(data)
        for i, scaler in enumerate(self.scalers):
            transformed_data[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
        return transformed_data

    def transform(self, data):
        transformed_data = np.zeros_like(data)
        for i, scaler in enumerate(self.scalers):
            transformed_data[:, i] = scaler.transform(data[:, i].reshape(-1, 1)).flatten()
        return transformed_data

def preprocess_and_save(data_dir, test_set, save_dir="./preprocessed_data", limit=None, downsample_ratio=10):
    """
    파일 단위 전처리 후 npy 저장. 다운샘플링 및 파일 수 제한 포함.
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset = BearingDataset(data_dir, test_set, transform=None, downsample_ratio=downsample_ratio)

    num_channels = 4
    num_files = len(dataset) if limit is None else min(limit, len(dataset))
    sample_length = dataset[0].shape[0]  # 다운샘플된 길이

    save_path = os.path.join(save_dir, f"{test_set}_processed.npy")
    processed_data = np.lib.format.open_memmap(save_path, mode='w+', dtype='float32',
                                               shape=(num_files, sample_length, num_channels))

    preprocessor = DataPreprocessor(num_channels=num_channels)

    all_data = []
    for i in range(num_files):
        all_data.append(dataset[i])
    all_data = np.vstack(all_data)
    preprocessor.fit_transform(all_data)

    for i in range(num_files):
        processed_data[i] = preprocessor.transform(dataset[i])
        print(f"Processed file {i + 1}/{num_files} for {test_set}")

    del processed_data
    print(f"{test_set} 데이터셋이 {save_path}에 저장되었습니다.")

class CompressedBearingDataset(Dataset):
    """
    전처리된 npy 파일을 로드하여 Torch Dataset 형식으로 제공
    """
    def __init__(self, npy_file):
        self.data = np.load(npy_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_compressed_dataloaders(npy_file, batch_size=32, shuffle=True, num_workers=4):
    dataset = CompressedBearingDataset(npy_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
