# training.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_transformer import AdvancedAutoInformerModel

def train_model(train_loader, model=None, num_epochs=2, device="cuda"):
    """
    주어진 DataLoader로 Transformer 모델을 학습
    기존 모델이 없으면 자동 초기화하며, MSE 손실 기준으로 AdamW + Cosine Scheduler 사용
    """
    if model is None:
        # 기존 모델이 없으면 새로운 모델을 초기화
        # 입력 차원 자동 설정
        input_dim = train_loader.dataset.data.shape[2]
        model = AdvancedAutoInformerModel(input_dim=input_dim, 
                                          model_dim=64,
                                          num_heads=4, 
                                          num_layers=2, 
                                          output_dim=4)
    model.to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # 학습 루프
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs in train_loader:
            inputs = inputs.to(device).float()
            optimizer.zero_grad()    # 기울기 초기화
            outputs = model(inputs)  # 모델에 입력 데이터 전달

            # 출력과 마지막 타임 스텝의 입력을 비교하여 손실 계산
            target = inputs[:, -1, :4].to(device)
            loss = criterion(outputs, target)
            loss.backward()          # 역전파 수행
            optimizer.step()         # 모델 파라미터 업데이트
            running_loss += loss.item()

        # 학습률 조정
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 학습된 모델 반환
    return model
