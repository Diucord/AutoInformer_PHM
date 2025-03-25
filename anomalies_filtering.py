# anomalies_filtering.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import platform
from concurrent.futures import ThreadPoolExecutor
import matplotlib.font_manager as fm
import matplotlib as mpl

class BearingDataset:
    """
    IMS Bearing Dataset을 2차원 배열로 로드하여 반환
    """
    def __init__(self, data_dir, test_set='1st_test'):
        self.data_dir = os.path.join(data_dir, test_set)
        self.file_list = sorted(os.listdir(self.data_dir))

        # 데이터셋에 따른 채널 구성
        self.num_channels = 8 if test_set == '1st_test' else 4

    def __len__(self):
        """
        데이터 파일 개수 반환
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        데이터 파일을 읽어 2차원 형태로 반환
        """
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = pd.read_csv(file_path, header=None, sep='\t').values

        # 1st_test의 경우 8채널 데이터를 4채널로 축소
        if self.num_channels == 8:
            data = np.column_stack([
                (data[:, 0] + data[:, 1]) / 2,  # 베어링 1
                (data[:, 2] + data[:, 3]) / 2,  # 베어링 2
                (data[:, 4] + data[:, 5]) / 2,  # 베어링 3
                (data[:, 6] + data[:, 7]) / 2   # 베어링 4
            ])

        return data  # 반환 형태: (타임스텝, 4채널)


def load_and_combine_data_in_batches(data_dir, datasets, cache_file, batch_size=100):
    """
    모든 데이터셋을 배치 단위로 병합하여 베어링별로 데이터를 분리하고 캐싱
    """
    if os.path.exists(cache_file):
        print(f"캐시된 데이터 로드 중: {cache_file}")
        return np.load(cache_file, allow_pickle=True).item()

    print("\n모든 데이터셋 병합 및 처리 시작...")
    num_bearings = 4  # 베어링 수
    combined_data = {f'bearing_{i+1}': [] for i in range(num_bearings)}

    for dataset_name in datasets:
        print(f"데이터셋 '{dataset_name}' 처리 중...")
        dataset_instance = BearingDataset(data_dir, dataset_name)
        num_samples = len(dataset_instance)

        def process_file(idx):
            return dataset_instance[idx]

        with ThreadPoolExecutor() as executor:
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch_data = list(executor.map(process_file, range(batch_start, batch_end)))
                batch_data = np.array(batch_data)

                # 베어링별로 데이터 추가
                for bearing in range(num_bearings):
                    combined_data[f'bearing_{bearing+1}'].append(batch_data[:, :, bearing])

    # 병합 및 저장
    for bearing in combined_data.keys():
        combined_data[bearing] = np.concatenate(combined_data[bearing], axis=0)

    np.save(cache_file, combined_data)
    print(f"병합된 데이터가 저장되었습니다: {cache_file}")
    return combined_data


def filter_outliers(combined_data, sigma_threshold=8, save_filtered_file=None):
    """
    8시그마 이상 데이터를 필터링하여 제거
    Args:
        combined_data (dict): 베어링별 데이터 딕셔너리
        sigma_threshold (int): 제거할 시그마 임계값
        save_filtered_file (str): 필터링된 데이터를 저장할 경로 (옵션)
    Returns:
        filtered_data (dict): 8시그마 이상 데이터를 제거한 필터링된 데이터
    """
    filtered_data = {}

    for bearing, data in combined_data.items():
        mean_value = np.mean(data)
        std_deviation = np.std(data)

        # 8시그마 이상 데이터만 필터링
        inlier_mask = (data <= mean_value + sigma_threshold * std_deviation) & \
                      (data >= mean_value - sigma_threshold * std_deviation)
        filtered_data[bearing] = data[inlier_mask]

        print(f"{bearing}: 원본 데이터 개수 = {len(data)}, 필터링된 데이터 개수 = {len(filtered_data[bearing])}")
        print(f"{bearing}: 제거된 데이터 개수 = {len(data) - len(filtered_data[bearing])}")

    # 필터링된 데이터 저장
    if save_filtered_file:
        np.save(save_filtered_file, filtered_data)
        print(f"필터링된 데이터가 저장되었습니다: {save_filtered_file}")

    print("\n8시그마 이상 데이터 제거 완료!")
    return filtered_data
