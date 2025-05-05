# anomaly_detection.py
import os
import csv
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.optim import lr_scheduler
from adabelief_pytorch import AdaBelief
import seaborn as sns

sns.set_theme(style="whitegrid")

class AnomalyDetector(nn.Module):
    """
    Autoencoder 기반의 이상치 탐지 모델 클래스
    입력 데이터를 인코딩하고 다시 디코딩하여 입력과 출력의 차이(reconstruction error) 계산
    """
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_autoencoder(
        self,
        data: TensorDataset,
        num_epochs: int = 20,
        batch_size: int = 32,
        lr: float = 0.001,
        model_path: str = "anomaly_detector.pth",
    ):
        """
        Autoencoder를 학습하고 모델 가중치를 저장합니다.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()

        self.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in data_loader:
                batch = batch[0]
                optimizer.zero_grad()
                outputs = self(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}")

        torch.save(self.state_dict(), model_path)
        print(f"모델 가중치 저장 완료: {model_path}")


def get_optimizer(model: nn.Module, lr: float, optimizer_choice: str):
    """
    모델 학습을 위한 옵티마이저를 설정합니다.
    """
    if optimizer_choice == "RAdam":
        return optim.RAdam(model.parameters(), lr=lr)
    elif optimizer_choice == "AdaBelief":
        return AdaBelief(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        return optim.Adam(model.parameters(), lr=lr)


def compare_methods(data: torch.Tensor):
    pca_results = pca_transform(data, n_components=4)
    lof_results = lof_anomaly_detection(data)
    print(f"PCA Result Shape: {pca_results.shape}")
    print(f"LOF Detected Anomalies: {lof_results.sum()}")

def compute_reconstruction_error(data_loader, model: nn.Module) -> np.ndarray:
    """
    배치 기반 Reconstruction Error 계산.
    """
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch[0]  # TensorDataset 형식에서 데이터 추출
            error = torch.mean((batch - model(batch)) ** 2, dim=1)
            errors.append(error.numpy())
    return np.concatenate(errors)

def compute_hourly_mean(data: np.ndarray, interval_minutes: int = 10) -> np.ndarray:
    """
    Reconstruction Error 데이터를 1시간 단위로 평균화.

    Parameters:
    - data (np.ndarray): Reconstruction Error 데이터.
    - interval_minutes (int): 시간 간격 (분 단위).

    Returns:
    - np.ndarray: 1시간 단위로 평균화된 Reconstruction Error 데이터.
    """
    samples_per_hour = 60 // interval_minutes
    num_hours = len(data) // samples_per_hour
    hourly_means = [
        np.mean(data[i * samples_per_hour:(i + 1) * samples_per_hour])
        for i in range(num_hours)
    ]
    # 나머지 데이터 처리
    if len(data) % samples_per_hour > 0:
        hourly_means.append(np.mean(data[num_hours * samples_per_hour:]))
    return np.array(hourly_means)



def pca_transform(data: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    """
    PCA를 사용해 데이터 차원을 축소합니다.
    """
    if data.ndim == 3:
        data = data.view(-1, data.shape[-1])  # 3D 데이터를 2D로 변환
    pca = PCA(n_components=n_components)
    return torch.tensor(pca.fit_transform(data.numpy()))


def lof_anomaly_detection(data: torch.Tensor, contamination: float = 0.1) -> torch.Tensor:
    """
    LOF (Local Outlier Factor)를 사용해 이상치를 탐지합니다.
    """
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_scores = -lof.fit_predict(data.numpy())
    return torch.tensor(lof_scores)


def save_anomaly_summary(
    reconstruction_error: np.ndarray,
    threshold: float,
    output_dir: str,
    file_name: str = "anomaly_summary.csv"
):
    """
    1시간 단위 Reconstruction Error 데이터를 기반으로 이상치 탐지 결과를 요약하여 저장.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    anomalies = reconstruction_error > threshold
    anomaly_ranges = []
    start = None
    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly and start is None:
            start = i
        elif not is_anomaly and start is not None:
            anomaly_ranges.append((start, i - 1))
            start = None
    if start is not None:
        anomaly_ranges.append((start, len(anomalies) - 1))

    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Start Hour", "End Hour", "Average Error"])
        for start, end in anomaly_ranges:
            avg_error = reconstruction_error[start:end + 1].mean()
            writer.writerow([start, end, avg_error])

    logging.info(f"Anomaly summary saved to {file_path}")
    return anomaly_ranges


def save_clustered_anomalies(
    anomaly_indices: np.ndarray,
    labels: np.ndarray,
    reconstruction_error: np.ndarray,
    output_dir: str,
    file_name: str = "clustered_anomalies.csv"
):
    """
    클러스터링 결과를 CSV로 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Index", "Cluster", "Reconstruction Error"])
        for idx, label, error in zip(anomaly_indices, labels, reconstruction_error[anomaly_indices]):
            writer.writerow([idx, label, error])

    print(f"Clustered anomalies saved to {file_path}")

def plot_top_n_anomalies(
    elapsed_time: np.ndarray,
    reconstruction_error: np.ndarray,
    threshold: float,
    top_n: int,
    output_dir: str,
    file_name: str = "top_n_anomalies.png"
):
    """
    상위 N개의 이상치에 대한 시각화를 저장합니다.
    """
    top_indices = np.argsort(reconstruction_error)[-top_n:][::-1]
    top_errors = reconstruction_error[top_indices]
    top_times = elapsed_time[top_indices]

    plt.figure(figsize=(12, 6))
    plt.plot(elapsed_time, reconstruction_error, label="Reconstruction Error", color="gray")
    plt.scatter(top_times, top_errors, color="red", label=f"Top {top_n} Anomalies", zorder=5)
    plt.axhline(y=threshold, color="blue", linestyle="--", label="Threshold")
    plt.title(f"Top {top_n} Anomalies")
    plt.xlabel("Elapsed Time (hours)")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, file_name)
    plt.savefig(plot_path)
    plt.show()
    print(f"Top {top_n} anomalies plot saved to {plot_path}")


def plot_clustered_anomalies(
    elapsed_time: np.ndarray,
    reconstruction_error: np.ndarray,
    threshold: float,
    output_dir: str,
    file_name: str = "clustered_anomalies.png",
    n_clusters: int = 3
):
    """
    시계열 이상치를 클러스터링하고 결과를 시각화합니다.
    """
    from sklearn.cluster import KMeans

    anomalies = reconstruction_error > threshold
    anomaly_indices = np.where(anomalies)[0]
    anomaly_errors = reconstruction_error[anomaly_indices]
    anomaly_times = elapsed_time[anomaly_indices].reshape(-1, 1)

    if len(anomaly_times) < n_clusters:
        print("Not enough anomalies to cluster. Skipping clustering visualization.")
        return

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(anomaly_times)

    plt.figure(figsize=(12, 6))
    plt.scatter(anomaly_times, anomaly_errors, c=labels, cmap="viridis", label="Clustered Anomalies")
    plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
    plt.title(f"Clustered Anomalies (n_clusters={n_clusters})")
    plt.xlabel("Elapsed Time (hours)")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, file_name)
    plt.savefig(plot_path)
    plt.show()
    print(f"Clustered anomalies plot saved to {plot_path}")


def plot_with_anomalies(
        elapsed_time: np.ndarray,
        reconstruction_error: np.ndarray,
        threshold: float,
        output_dir: str,
        file_name: str = "highlighted_anomalies_plot.png"
):
    """
    전체 데이터에서 이상치 구간을 강조한 시각화를 생성하고 저장합니다.
    """
    anomalies = reconstruction_error > threshold
    anomaly_indices = np.where(anomalies)[0]

    plt.figure(figsize=(12, 6))
    plt.plot(elapsed_time, reconstruction_error, label="Reconstruction Error", color="gray")
    plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
    plt.fill_between(
        elapsed_time,
        reconstruction_error,
        threshold,
        where=anomalies,
        color="orange",
        alpha=0.3,
        label="Anomalies"
    )
    plt.scatter(
        elapsed_time[anomaly_indices],
        reconstruction_error[anomaly_indices],
        color="red",
        label="Anomaly Points",
        zorder=5
    )
    plt.title("Reconstruction Error with Highlighted Anomalies")
    plt.xlabel("Elapsed Time (hours)")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, file_name)
    plt.savefig(plot_path)
    plt.show()
    print(f"Highlighted anomalies plot saved to {plot_path}")