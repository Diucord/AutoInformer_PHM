# evaluation.py
import torch
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from data_preprocessing import create_compressed_dataloaders
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def basic_data_analysis_plot(data, interval='1H'):
    """
    Step 1: Basic data analysis plot to understand initial patterns.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(data, label="Raw VSF Data")
    plt.title("Basic Data Analysis - Raw Data over Time")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


def pca_clustering_plot(data, n_clusters=2):
    """
    Step 2: PCA and clustering plot to define normal and anomalous states.
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(reduced_data)

    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", marker="o")
    plt.title("PCA and KMeans Clustering for Anomaly Detection")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()


def evaluate_model_with_rul(model, npy_file, batch_size=32, threshold=0.1, consecutive_threshold_steps=10,
                            device=torch.device("cpu")):
    """
    고장 예측 및 잔여 수명(RUL)을 출력하는 평가 함수
    """
    print(f"Evaluation Device: {device}")
    dataloader = create_compressed_dataloaders(npy_file=npy_file, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    actuals, predictions, warnings = [], [], []
    consecutive_exceedances = 0  # 연속 임계 초과 카운터 초기화
    predicted_failure_date = None  # 예상 고장 날짜 초기화
    estimated_rul = None  # 잔여 수명 초기화
    warning_levels = []  # 경고 수준 저장

    # 데이터셋에 따른 시간 간격, 고장 베어링 및 고장 유형 설정
    if '1st_test' in npy_file:
        time_interval = 10 if len(actuals) >= 43 else 5  # 첫 43개 파일은 5분 간격, 이후는 10분 간격
        target_bearing = 3  # 고장 베어링 설정
        failure_type = "Inner Ring Fault in Bearing 3"
    elif '2nd_test' in npy_file:
        time_interval = 10  # 10분 간격
        target_bearing = 1  # 고장 베어링 설정
        failure_type = "Outer Ring Fault in Bearing 1"
    elif '3rd_test' in npy_file:
        time_interval = 10  # 10분 간격
        target_bearing = 3  # 고장 베어링 설정
        failure_type = "Outer Ring Fault in Bearing 3"
    else:
        time_interval = 10  # 기본값
        target_bearing = None
        failure_type = "Unknown Fault"

    print(f"Evaluating dataset: {npy_file} for {failure_type}")

    # 현재 날짜를 기준으로 고장 예상 날짜를 계산
    current_date = datetime.now()

    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device).float()
            outputs = model(inputs)

            # 선택한 베어링의 실제 값과 예측 값만 가져옴
            actual_value = inputs[:, -1, target_bearing - 1].cpu().numpy()  # 베어링 인덱스에 맞게 조정
            predicted_value = outputs[:, target_bearing - 1].cpu().numpy()  # 베어링 인덱스에 맞게 조정

            actuals.append(inputs[:, -1, :].cpu().numpy())
            predictions.append(outputs.cpu().numpy())

            # 예측 오차 계산
            error = np.abs(outputs.cpu().numpy() - inputs[:, -1, :].cpu().numpy())
            exceed_threshold = error > threshold

            if np.any(exceed_threshold):
                consecutive_exceedances += 1
                warnings.append(True)
                warning_level = "High" if consecutive_exceedances >= consecutive_threshold_steps else "Medium"
                warning_levels.append(warning_level)
            else:
                consecutive_exceedances = 0
                warnings.append(False)
                warning_levels.append("Low")

            # 연속적인 임계 초과 발생 시 고장 시점 예상
            if consecutive_exceedances >= consecutive_threshold_steps:
                estimated_rul = (len(dataloader) - idx) * batch_size  # 잔여 수명 계산
                predicted_failure_date = current_date + timedelta(minutes=estimated_rul)  # 남은 시간 추정

                print("\n=== 고장 예측 ===")
                print(f"예상 잔여 수명: 약 {estimated_rul} 분")
                print(f"현재 경고 수준: {warning_level}")
                print("경고: 연속 임계값 초과로 고장 가능성이 높습니다.")
                break

    # 결과 데이터 시각화
    actuals = np.concatenate(actuals, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    # 결과 형상 맞춤
    if actuals.shape != predictions.shape:
        min_len = min(actuals.shape[0], predictions.shape[0])
        actuals, predictions = actuals[:min_len], predictions[:min_len]

    mse, mae = mean_squared_error(actuals, predictions), mean_absolute_error(actuals, predictions)
    print(f"\n전체 MSE: {mse}, 전체 MAE: {mae}")
    print(f"총 경고 횟수: {sum(warnings)}회")

    return actuals, predictions, warnings, warning_levels, predicted_failure_date, estimated_rul


def plot_results_with_warnings(actuals, predictions, warnings=None, warning_levels=None):
    """
    실제 값과 예측 값, 경고 수준을 시각화하며 텍스트 설명을 추가
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label="Actual", color="blue")
    plt.plot(predictions, label="Predicted", color="orange")

    if warnings is not None:
        warning_indices = np.where(warnings)[0]
        warning_colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'red'}

        for idx in warning_indices:
            level = warning_levels[idx]

            # 다차원인 경우 첫 번째 값을 선택하여 표시
            actual_value = actuals[idx][0] if actuals.ndim > 1 else actuals[idx]
            plt.axvline(x=idx, color=warning_colors[level], linestyle='--', label=f"Warning: {level}")

            try:
                plt.text(idx, actual_value, f"{level} Warning", color=warning_colors[level], rotation=90,
                         verticalalignment='bottom')
            except IndexError:
                print(f"Warning: Index {idx} is out of bounds for text placement.")

    plt.xlabel("Time Steps")
    plt.ylabel("Vibration Amplitude")
    plt.legend()
    plt.title("Vibration Prediction with Warning Levels")
    plt.show()
