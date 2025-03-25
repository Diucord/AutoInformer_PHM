# main.py
import os
from training import train_model
from evaluation import evaluate_model, plot_results
from result_output import save_results, load_and_plot_results
from data_preprocessing import preprocess_and_save, create_compressed_dataloaders

if __name__ == '__main__':
    # 데이터 디렉토리와 결과 저장 디렉토리
    data_dir = "./IMS"
    output_dir = "./cross_validation_results"
    preprocessed_dir = "./preprocessed_data"

    # 0. 전처리가 완료되지 않은 경우에만 전처리 수행
    for test_set in ['1st_test', '2nd_test', '3rd_test']:
        npy_file = os.path.join(preprocessed_dir, f"{test_set}_processed.npy")
        if os.path.exists(npy_file):
            print(f"{test_set} 데이터셋의 전처리 파일이 이미 존재합니다: {npy_file}")
            if test_set == '3rd_test':
                break  # 3번째 데이터셋까지 확인되면 반복 종료
            continue
        print(f"Preprocessing and saving {test_set} dataset...")
        preprocess_and_save(data_dir=data_dir, test_set=test_set, save_dir=preprocessed_dir)

    # 1. 각 데이터셋별로 모델을 학습하고 교차 검증
    for train_set in ['1st_test', '2nd_test', '3rd_test']:
        print(f"\n=== Training model on {train_set} dataset ===")
        npy_file = os.path.join(preprocessed_dir, f"{train_set}_processed.npy")
        train_loader = create_compressed_dataloaders(npy_file=npy_file, batch_size=128, shuffle=True, num_workers=4)

        # 모델 학습
        trained_model = train_model(train_loader)

        # 2. 교차 데이터셋 검증 수행
        for test_set in ['1st_test', '2nd_test', '3rd_test']:
            if train_set != test_set:
                print(f"Evaluating model trained on {train_set} using {test_set} as test set")

                test_npy_file = os.path.join(preprocessed_dir, f"{test_set}_processed.npy")
                test_loader = create_compressed_dataloaders(npy_file=test_npy_file, batch_size=128, shuffle=False, num_workers=4)

                actuals, predictions = evaluate_model(trained_model, test_loader)

                result_dir = os.path.join(output_dir, f"{train_set}_evaluated_on_{test_set}")
                os.makedirs(result_dir, exist_ok=True)
                save_results(actuals, predictions, output_dir=result_dir)

                plot_results(actuals, predictions)
