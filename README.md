# Bearing Failure Prediction with Transformer

이 프로젝트는 IMS Bearing Dataset을 기반으로 한 시계열 예측 기반 베어링 고장 조기 탐지 시스템입니다.  
Transformer 기반 모델을 활용하여 센서 진동 데이터를 예측하고, 이상 탐지 및 잔여 수명 예측(RUL)을 수행합니다.

## 🔍 주요 기능
- 센서 데이터 전처리 (정규화, 압축 등)
- Trend / Seasonality 분해 + Sparse Attention 기반 모델
- Reconstruction Error 기반 이상 탐지
- AUC, F1, Precision 등 정량 평가
- 고장 경고 시각화 + RUL 예측

## 🧪 모델 성능
- 평균 AUC: 1.00
- 평균 F1-score: 1.00
- 평균 지연시간 예측 오차: ±10분 이내 (실험 기준)

## 🛠 실행 방법
```bash
python main.py
