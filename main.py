# main.py
import os
import json
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from training import train_model
from evaluation import evaluate_model_with_rul, plot_results_with_warnings, compute_classification_metrics
from result_output import save_results
from data_preprocessing import preprocess_and_save, create_compressed_dataloaders
from anomaly_detection import compute_hourly_mean, save_anomaly_summary
from model_transformer import AdvancedAutoInformerModel


def model_path(train_set):
    return os.path.join("saved_models", f"{train_set}_model.pth")


if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)

    data_dir = "./IMS"
    preprocessed_dir = "./preprocessed_data"
    output_dir = "./predictive_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    device = config.get("device", "cpu")

    for test_set in ['1st_test', '2nd_test', '3rd_test']:
        npy_file = os.path.join(preprocessed_dir, f"{test_set}_processed.npy")
        if not os.path.exists(npy_file):
            preprocess_and_save(data_dir=data_dir, test_set=test_set,
                                save_dir=preprocessed_dir,
                                limit=config.get("limit_files"),
                                downsample_ratio=config.get("downsample_ratio", 10))
        else:
            print(f"{test_set} 데이터셋의 전처리 파일이 이미 존재합니다: {npy_file}")

    for train_set in ['1st_test', '2nd_test', '3rd_test']:
        print(f"\n=== Training or Loading model on {train_set} dataset ===")
        train_file = os.path.join(preprocessed_dir, f"{train_set}_processed.npy")
        train_loader = create_compressed_dataloaders(
            npy_file=train_file,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4
        )

        model_file = model_path(train_set)
        if os.path.exists(model_file):
            print(f"모델 불러오는 중: {model_file}")
            input_dim = train_loader.dataset.data.shape[2]
            model = AdvancedAutoInformerModel(input_dim=input_dim)
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.to(device)
        else:
            model = train_model(train_loader, num_epochs=config["num_epochs"], device=device)
            torch.save(model.state_dict(), model_file)
            print(f"모델 저장 완료: {model_file}")

        for test_set in ['1st_test', '2nd_test', '3rd_test']:
            if train_set == test_set:
                continue

            print(f"\nEvaluating model trained on {train_set} using {test_set} as test set")
            test_file = os.path.join(preprocessed_dir, f"{test_set}_processed.npy")
            result_dir = os.path.join(output_dir, f"{train_set}_to_{test_set}")
            os.makedirs(result_dir, exist_ok=True)

            actuals, predictions, warnings, levels, fail_date, rul = evaluate_model_with_rul(
                model,
                npy_file=test_file,
                batch_size=config["batch_size"],
                threshold=config["threshold"],
                consecutive_threshold_steps=config.get("consecutive_threshold_steps", 10)
            )

            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            print(f"\n=== 예측 정확도 평가 ({train_set} → {test_set}) ===")
            print(f"MSE: {mse:.6f}")
            print(f"MAE: {mae:.6f}")

            prediction_errors = np.mean((actuals - predictions) ** 2, axis=1)
            hourly_errors = compute_hourly_mean(prediction_errors, interval_minutes=10)

            anomaly_ranges = save_anomaly_summary(
                reconstruction_error=hourly_errors,
                threshold=config["threshold"],
                output_dir=result_dir,
                file_name=f"anomaly_summary_{train_set}_to_{test_set}.csv"
            )

            ground_truth = np.zeros_like(hourly_errors)
            for start, end in anomaly_ranges:
                ground_truth[start:end+1] = 1

            metrics = compute_classification_metrics(ground_truth, hourly_errors, threshold=config["threshold"])
            print(f"\n=== 이상 탐지 정량 평가 ({train_set} → {test_set}) ===")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            save_results(actuals, predictions, output_dir=result_dir)
            plot_results_with_warnings(actuals, predictions, warnings, levels)

            print(f"\n=== 예측 기반 고장 탐지 결과 ({train_set} → {test_set}) ===")
            if fail_date:
                print(f"예상 고장 시점: {fail_date.strftime('%Y-%m-%d %H:%M')}")
                print(f"남은 예측 수명 (분): {rul}")
            else:
                print("예상 고장 징후 없음.")
