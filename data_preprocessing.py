# data_preprocessing.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class BearingDataset(Dataset):
    """
    IMS Bearing Dataset을 Torch Dataset 형식으로 변환하여 로드
    """
    def __init__(self, data_dir, test_set='1st_test', transform=None):
        self.data_dir = os.path.join(data_dir, test_set)
        self.file_list = sorted(os.listdir(self.data_dir))
        self.transform = transform

        # 데이터셋에 따른 채널 구성
        self.num_channels = 8 if test_set == '1st_test' else 4

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = pd.read_csv(file_path, header=None, sep='\t').values

        # 데이터셋 1의 경우, 8채널 데이터를 4채널로 축소 (각 베어링의 두 센서 데이터를 평균)
        if self.num_channels == 8:
            data = np.column_stack([
                (data[:, 0] + data[:, 1]) / 2,  # 베어링 1의 x, y 센서 평균
                (data[:, 2] + data[:, 3]) / 2,  # 베어링 2의 x, y 센서 평균
                (data[:, 4] + data[:, 5]) / 2,  # 베어링 3의 x, y 센서 평균
                (data[:, 6] + data[:, 7]) / 2   # 베어링 4의 x, y 센서 평균
            ])

        if self.transform:
            data = self.transform(data)
        return data

class DataPreprocessor:
    """
    데이터 전처리 클래스
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

def preprocess_and_save(data_dir, test_set, save_dir="./preprocessed_data", limit=None):
    """
    데이터셋을 파일 단위로 전처리하고 npy 파일로 저장
    limit: 처리할 파일 수를 제한(테스트용)
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset = BearingDataset(data_dir, test_set, transform=None)

    # 모든 데이터셋을 4채널로 맞추기 위해 4채널로 설정
    num_channels = 4
    num_files = len(dataset) if limit is None else min(limit, len(dataset))

    # npy 파일을 메모리 맵핑 방식으로 생성
    save_path = os.path.join(save_dir, f"{test_set}_processed.npy")
    processed_data = np.lib.format.open_memmap(save_path, mode='w+', dtype='float32',
                                               shape=(num_files, 20480, num_channels))

    # 데이터 전처리 클래스 초기화
    preprocessor = DataPreprocessor(num_channels=num_channels)

    # 전체 데이터 스케일러 학습
    all_data = []
    for i in range(num_files):
        file_data = dataset[i]
        all_data.append(file_data)
    all_data = np.vstack(all_data)

    # 전체 데이터로 스케일러 학습
    preprocessor.fit_transform(all_data)

    # 개별 파일 전처리 및 저장
    for i in range(num_files):
        file_data = dataset[i]

        # 스케일링 후 메모리 맵 파일에 저장
        processed_data[i] = preprocessor.transform(file_data)
        print(f"Processed file {i + 1}/{num_files} for {test_set}")

    # 저장 완료 후 메모리 맵핑 해제
    del processed_data
    print(f"{test_set} 데이터셋이 {save_path}에 저장되었습니다.")

class CompressedBearingDataset(Dataset):
    """
    전처리된 npy 파일을 로드하여 Torch Dataset 형식으로 제공
    """
    def __init__(self, npy_file):
        # npy 파일로부터 데이터를 메모리에 로드
        self.data = np.load(npy_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_compressed_dataloaders(npy_file, batch_size=32, shuffle=True, num_workers=4):
    """
    압축된 npy 파일로부터 데이터 로더를 생성합니다.
    """
    dataset = CompressedBearingDataset(npy_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
