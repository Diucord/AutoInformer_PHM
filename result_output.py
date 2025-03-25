# result_output.py
import numpy as np
import os
from evaluation import plot_results_with_warnings

def save_results(actuals, predictions, output_dir="results"):
    """
    실제 값과 예측 값을 저장합니다.
    """
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 실제 값과 예측 값 저장
    np.save(f"{output_dir}/actuals.npy", actuals)
    np.save(f"{output_dir}/predictions.npy", predictions)
    print(f"결과가 {output_dir}에 저장되었습니다")

def load_and_plot_results(actual_path, prediction_path):
    """
    저장된 결과를 로드하고 시각화합니다.
    """
    # 실제 값과 예측 값 로드
    actuals = np.load(actual_path)
    predictions = np.load(prediction_path)

    # warnings 및 warning_levels가 저장되지 않은 경우 빈 리스트를 제공합니다.
    plot_results_with_warnings(actuals, predictions, warnings=[], warning_levels=[])
