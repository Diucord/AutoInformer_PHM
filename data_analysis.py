import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_analyze_data(file_path):
    """
    데이터 파일을 로드하고 기초 통계를 출력하고 시계열 그래프를 그려 이상치를 시각적으로 탐색
    """
    # 데이터 로드
    data = np.load(file_path)
    data_df = pd.DataFrame(data)

    # 기초 통계량 출력
    print("기초 통계량:")
    print(data_df.describe())

    # 시계열 그래프를 통해 데이터 분포 시각화
    plt.figure(figsize=(14, 6))
    plt.plot(data_df, label="Original Data")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Time Series Data Analysis")
    plt.legend()
    plt.show()

    return data
