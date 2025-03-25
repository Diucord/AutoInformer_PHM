# model_transformer.py
import torch
import torch.nn as nn
import math

class MultiScaleTrendSeasonalityDecomposition(nn.Module):
    """
    트렌드 및 계절성 분해를 위한 모듈로, 다중 주기의 트렌드 및 계절성을 학습
    """
    def __init__(self, model_dim, scales=[1, 3, 5]):
        super(MultiScaleTrendSeasonalityDecomposition, self).__init__()

        # 주기마다 Linear Layer를 추가하여 트렌드와 계절성 학습
        self.trend_layers = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in scales])
        self.seasonality_layers = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in scales])

    def forward(self, x):
        # 트렌드와 계절성을 각 스케일에 따라 학습
        trend = sum(layer(x) for layer in self.trend_layers)
        seasonality = sum(layer(x) for layer in self.seasonality_layers)
        return trend, seasonality

class BlockSparseSelfAttention(nn.Module):
    """
    블록 기반 Sparse Self-Attention 모듈
    메모리 절약을 위해 일정 블록 단위로만 Self-Attention을 수행
    """
    def __init__(self, model_dim, num_heads, block_size=10):
        super(BlockSparseSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)
        self.block_size = block_size

    def forward(self, x):
        output_blocks = []
        for i in range(0, x.size(1), self.block_size):
            block_x = x[:, i:i + self.block_size].transpose(0, 1)  # transpose for attention
            attn_output, _ = self.attention(block_x, block_x, block_x)
            output_blocks.append(attn_output.transpose(0, 1))  # transpose back
        return torch.cat(output_blocks, dim=1)


class AdaptivePositionalEncoding(nn.Module):
    """
    트렌드와 계절성을 반영한 고도화된 Transformer 기반 시계열 예측
    """
    def __init__(self, model_dim, max_len=5000):
        super(AdaptivePositionalEncoding, self).__init__()
        self.max_len = max_len
        self.model_dim = model_dim

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device  # 입력 텐서의 장치를 가져옵니다.

        # pos_encoding과 position을 GPU로 이동
        pos_encoding = torch.zeros(seq_len, self.model_dim, device=device)  # GPU로 이동
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)

        # div_term을 GPU로 이동
        div_term = torch.exp(torch.arange(0, self.model_dim, 2, dtype=torch.float, device=device) * (
                    -math.log(10000.0) / self.model_dim))

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding.unsqueeze(0) + x


class MultiScaleFeatureExtractor(nn.Module):
    """
    여러 스케일의 입력 특징을 추출하는 모듈
    """
    def __init__(self, input_dim, model_dim, scales=[1, 3, 5]):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.scale_layers = nn.ModuleList([
            nn.Conv1d(input_dim, model_dim, kernel_size=scale, padding=scale // 2) for scale in scales])

    def forward(self, x):
        # 배치 차원 교환
        x = x.permute(0, 2, 1)
        features = sum(layer(x) for layer in self.scale_layers)
        return features.permute(0, 2, 1)

class AdvancedAutoInformerModel(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, output_dim=4, dropout=0.1, max_len=5000):
        """
        output_dim을 1로 설정하여 각 베어링에 대한 예측 결과를 독립적으로 생성
        """
        super(AdvancedAutoInformerModel, self).__init__()
        self.feature_extractor = MultiScaleFeatureExtractor(input_dim, model_dim)
        self.positional_encoding = AdaptivePositionalEncoding(model_dim, max_len)
        self.decomposition = MultiScaleTrendSeasonalityDecomposition(model_dim)
        self.layers = nn.ModuleList([BlockSparseSelfAttention(model_dim, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, output_dim) # 각 베어링의 최종 출력은 독립적으로 계산
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.positional_encoding(x)
        trend, seasonality = self.decomposition(x)

        # 트렌드와 계절성 정보 추가
        x = x + trend + seasonality
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1, :]

        # 마지막 타임스텝 출력
        return self.fc_out(x)